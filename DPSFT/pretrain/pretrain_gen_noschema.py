"""Generate text samples using pretrained Hugging Face models without a JSON schema input.

Generation pipeline:
- Single iteration (default): Generate abstracts using a bare 'gen' prompt
- Multi-iteration: Generate → Filter by voting → Refine with 'variation' prompt
- After all generation, assign schema attributes to final outputs via Gemini API

The key difference from pretrain_gen.py: no JSON schema is used as input feature.
Schema values are assigned *post-hoc* using the annotation Gemini pipeline.
"""

import argparse
import json
import logging
import os
import os.path as osp
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Add DPSFT directory to path (parent of pretrain/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dpsft_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(dpsft_dir)
sys.path.insert(0, dpsft_dir)

from utils.pretrain_utils import get_pretrain_generation_prompt_dict_noschema
from dotenv import load_dotenv

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
)

# Load environment variables from repo root
load_dotenv(os.path.join(repo_root, '.env'))
hf_token = os.getenv("HF_LOGIN_STR")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(file_path):
  """Load JSONL file into list of dicts."""
  data = []
  with open(file_path, 'r') as f:
    for line in f:
      data.append(json.loads(line.strip()))
  return data


def save_jsonl(data, file_path):
  """Save list of dicts to JSONL file."""
  with open(file_path, 'w') as f:
    for item in data:
      f.write(json.dumps(item, ensure_ascii=False) + '\n')
  logging.info(f"Saved {len(data)} samples to: {file_path}")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def run_evaluation(generated_file, real_data_path, output_dir, iteration, device='cuda:0', real_text_column='abstract'):
  """Run evaluation if eval_utils is available."""
  try:
    from pretrain.eval_func import evaluate_iteration
    results = evaluate_iteration(
      generated_file, real_data_path, output_dir, iteration,
      gen_text_column='generated_text', real_text_column=real_text_column, device=device
    )
    return results
  except ImportError as e:
    logging.warning(f"eval_func not found, skipping evaluation: {e}")
    return None


def filter_by_voting(generated_data, real_data_path, top_k,
                     embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                     device='cuda', rho=None, text_column='abstract'):
  """Filter generated data by voting: each real sample votes for its nearest generated sample."""
  from sentence_transformers import SentenceTransformer

  logging.info(f"Loading real data from: {real_data_path}")
  if real_data_path.endswith('.jsonl'):
    data = load_jsonl(real_data_path)
    real_texts = [item[text_column] for item in data]
  elif real_data_path.endswith('.csv'):
    df = pd.read_csv(real_data_path)
    real_texts = df[text_column].tolist()
  else:
    raise ValueError(f"Unsupported file format: {real_data_path}")

  logging.info(f"Loaded {len(real_texts)} real samples")

  generated_texts = [item['generated_text'] for item in generated_data]
  logging.info(f"Computing embeddings for {len(generated_texts)} generated samples")

  embedding_model = SentenceTransformer(embedding_model_name, device=device)
  encode_batch_size = 256 if 'MiniLM' in embedding_model_name else 128

  logging.info(f"Computing real embeddings with batch_size={encode_batch_size}...")
  real_embeddings = embedding_model.encode(
    real_texts, batch_size=encode_batch_size, show_progress_bar=True,
    device=device, convert_to_numpy=True, normalize_embeddings=True
  )

  logging.info(f"Computing generated embeddings with batch_size={encode_batch_size}...")
  generated_embeddings = embedding_model.encode(
    generated_texts, batch_size=encode_batch_size, show_progress_bar=True,
    device=device, convert_to_numpy=True, normalize_embeddings=True
  )

  # Use FAISS for fast nearest neighbor search
  logging.info("Building FAISS index and computing nearest neighbors (voting)...")
  try:
    import faiss

    # Ensure embeddings are float32 and contiguous (FAISS requirement)
    generated_embeddings = np.ascontiguousarray(generated_embeddings.astype(np.float32))
    real_embeddings = np.ascontiguousarray(real_embeddings.astype(np.float32))

    d = generated_embeddings.shape[1]  # Dimension

    # Use IndexFlatIP for inner product (cosine similarity since embeddings are normalized)
    index = faiss.IndexFlatIP(d)

    # Try to use GPU if available
    if device.startswith('cuda'):
      try:
        gpu_id = int(device.split(':')[-1]) if ':' in device else 0
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        logging.info(f"Using FAISS GPU index on {device}")
      except Exception as e:
        logging.warning(f"Failed to move FAISS to GPU: {e}. Using CPU.")

    # Add generated embeddings to index
    index.add(generated_embeddings)

    # Search for nearest neighbor for each real embedding
    # k=1 means we only want the top-1 nearest neighbor
    similarities, nearest_indices = index.search(real_embeddings, k=1)
    nearest_indices = nearest_indices.flatten()  # Shape: (n_real,)

  except ImportError:
    logging.warning("FAISS not available, falling back to numpy implementation")
    # Fallback to numpy (original implementation)
    nearest_indices = []
    chunk_size = 1000

    for i in range(0, len(real_embeddings), chunk_size):
      chunk_real = real_embeddings[i:i+chunk_size]
      similarities = np.dot(chunk_real, generated_embeddings.T)
      chunk_nearest = np.argmax(similarities, axis=1)
      nearest_indices.extend(chunk_nearest)

    nearest_indices = np.array(nearest_indices)

  # Count votes
  vote_counts = np.bincount(nearest_indices, minlength=len(generated_data))

  # Add DP noise if rho is specified
  if rho:
    vote_counts = vote_counts.astype(np.float64)
    vote_counts += np.sqrt(1/(2*rho)) * np.random.randn(*vote_counts.shape)

  # Select top-k samples with most votes
  top_k_indices = np.argsort(vote_counts)[::-1][:top_k]

  logging.info(f"Vote distribution - Max: {vote_counts.max()}, Min: {vote_counts.min()}, "
               f"Mean: {vote_counts.mean():.2f}, Median: {np.median(vote_counts):.2f}")
  logging.info(f"Selected top-{top_k} samples with votes >= {vote_counts[top_k_indices[-1]]}")

  return [generated_data[i] for i in top_k_indices]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batch(model, tokenizer, prompts, args, device, terminators):
  """Generate text for a batch of prompts."""
  batch = tokenizer(
    prompts,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=args.prompt_len,
    add_special_tokens=False
  ).to(device)

  input_len = batch['input_ids'].shape[1]

  with torch.no_grad():
    outputs = model.generate(
      **batch,
      max_new_tokens=args.seq_len,
      pad_token_id=tokenizer.pad_token_id,
      eos_token_id=terminators,
      do_sample=True,
      top_p=args.top_p,
      top_k=args.top_k,
      temperature=args.temperature,
    )

  generated_tokens = outputs[:, input_len:]
  return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def run_generation(model, tokenizer, input_data, args, device, terminators,
                   prompt_type='gen',
                   system_prompt="You are a helpful assistant. Follow the instructions strictly."):
  """
  Run generation for one iteration (no-schema version).

  For 'gen':       input_data items need no fields (generates freely).
  For 'variation': input_data items must have 'generated_text'.
  """
  logging.info(f"Running generation with prompt type: {prompt_type}, {len(input_data)} samples")

  prompt_dict = get_pretrain_generation_prompt_dict_noschema(
    f'{args.prompt_str}_generation', prompt_type
  )
  user_template = prompt_dict['prompt']

  all_results = []

  for i in tqdm(range(0, len(input_data), args.bs), desc="Generating"):
    batch_data = input_data[i : i + args.bs]

    formatted_prompts = []
    for item in batch_data:
      if prompt_type == 'gen':
        messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_template},
        ]
      else:  # variation
        messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_template.format(text=item['generated_text'])},
        ]

      formatted_prompts.append(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      )

    decoded_texts = generate_batch(model, tokenizer, formatted_prompts, args, device, terminators)

    for item, out in zip(batch_data, decoded_texts):
      all_results.append({'generated_text': out.strip()})

  logging.info(f"Generation complete. Generated {len(all_results)} samples.")
  return all_results


# ---------------------------------------------------------------------------
# Post-hoc schema assignment via Gemini API
# ---------------------------------------------------------------------------

_GEMINI_MAX_RETRIES = 5
_GEMINI_MODEL = 'gemini-2.5-flash-lite-preview-09-2025'


def _infer_schema_single(abstract_text, client, prompt_template):
  """Call Gemini API to assign schema attributes to a single abstract."""
  for attempt in range(_GEMINI_MAX_RETRIES):
    try:
      full_prompt = prompt_template.format(abstract_text=abstract_text)
      response = client.models.generate_content(model=_GEMINI_MODEL, contents=full_prompt)
      return response.text.strip()
    except Exception as e:
      wait_time = (2 ** attempt) + random.uniform(0, 1)
      logging.warning(
        f"Gemini API call failed: {e}. Retrying in {wait_time:.2f}s "
        f"(attempt {attempt + 1}/{_GEMINI_MAX_RETRIES})"
      )
      time.sleep(wait_time)
  logging.error("Schema inference failed after all retries.")
  return "GENERATION_ERROR"


def _clean_schema(val):
  """Strip markdown fences and return compact JSON string."""
  if not val or val == 'GENERATION_ERROR':
    return val
  val = val.strip()
  if val.startswith('```'):
    val = re.sub(r'^```(?:json)?\s*', '', val)
    val = re.sub(r'\s*```$', '', val)
  val = val.strip()
  try:
    parsed = json.loads(val)
    return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
  except json.JSONDecodeError:
    return val


def assign_schemas(generated_data, schema_name, schema_prompt_file, max_workers=8):
  """
  Assign schema attributes to each item in generated_data using the Gemini API.

  Reads:
    - annotation/schema/{schema_name}.txt   attribute schema
    - {schema_prompt_file}                  extraction prompt template

  Adds a 'schema' field (compact JSON string) to each item in-place and returns
  the updated list.
  """
  from google import genai as google_genai

  google_api_key = os.getenv("GOOGLE_API_KEY")
  if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

  client = google_genai.Client(api_key=google_api_key)

  # Load schema file
  schema_path = osp.join(repo_root, 'annotation', 'schema', f'{schema_name}.txt')
  with open(schema_path, 'r') as f:
    schema_content = f.read()
  logging.info(f"Loaded schema from: {schema_path}")

  # Load extraction prompt and embed schema
  with open(schema_prompt_file, 'r') as f:
    prompt_template_raw = f.read()

  # The prompt template uses {schema} and {abstract_text}
  prompt_template = prompt_template_raw.format(
    schema=schema_content, abstract_text='{abstract_text}'
  )

  abstracts = [item['generated_text'] for item in generated_data]
  infer_func = partial(_infer_schema_single, client=client, prompt_template=prompt_template)

  logging.info(
    f"Assigning schema to {len(abstracts)} samples using up to {max_workers} workers..."
  )
  t_start = time.time()
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    raw_schemas = list(tqdm(executor.map(infer_func, abstracts), total=len(abstracts)))

  for item, raw_schema in zip(generated_data, raw_schemas):
    item['schema'] = _clean_schema(raw_schema)

  elapsed = time.time() - t_start
  logging.info(f"Schema assignment complete in {elapsed:.2f}s.")
  return generated_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(
    description='Generate synthetic abstracts without schema input, then assign schema post-hoc.'
  )

  # Model arguments
  parser.add_argument('--device', '-d', type=int, default=0)
  parser.add_argument('--model_name_or_path', '-m', type=str, required=True,
                      help='HuggingFace Instruct model name or path')
  parser.add_argument('--rho', '-rho', type=float, default=0.0,
                      help='Privacy budget for noisy voting filter')

  # Data arguments
  parser.add_argument('--output_dir', '-o', type=str, required=True,
                      help='Output directory')
  parser.add_argument('--out_file', '-out', type=str, required=True,
                      help='Output filename (JSONL)')
  parser.add_argument('--prompt_str', '-ps', type=str, required=True,
                      help='Prompt template string (e.g., biorxiv_noexample)')

  # Generation arguments
  parser.add_argument('--prompt_len', '-pl', type=int, default=300,
                      help='Max input length')
  parser.add_argument('--seq_len', '-sl', type=int, default=512,
                      help='Max new tokens to generate')
  parser.add_argument('--bs', '-bs', type=int, default=64,
                      help='Batch size')
  parser.add_argument('--n_gen', '-n_gen', type=int, default=10000,
                      help='Number of final samples to produce')
  parser.add_argument('--top_p', '-tp', type=float, default=0.95)
  parser.add_argument('--top_k', '-tk', type=int, default=0)
  parser.add_argument('--temperature', '-temp', type=float, default=1.0)

  # Iterative refinement arguments
  parser.add_argument('--num_iterations', '-ni', type=int, default=0,
                      help='Number of iterations (0 = single pass, no filtering)')
  parser.add_argument('--real_data_path', '-rd', type=str, default=None,
                      help='Path to real data for voting-based filtering')
  parser.add_argument('--variation_rate', '-vr', type=int, default=1,
                      help='Generate n_gen * variation_rate samples per iteration, then filter to n_gen')
  parser.add_argument('--embedding_model', '-em', type=str,
                      default='sentence-transformers/all-mpnet-base-v2',
                      help='Embedding model for voting filter')
  parser.add_argument('--text_column', '-tc', type=str, default='abstract',
                      help='Text column in real data file')
  parser.add_argument('--evaluate', action='store_true',
                      help='Run evaluation after each iteration (requires real_data_path)')

  # Schema assignment arguments (post-hoc)
  parser.add_argument('--schema_name', '-sn', type=str, default="biorxiv_noexample",
                      help='Schema file name (without .txt) in annotation/schema/. '
                           'If omitted, schema assignment is skipped.')
  parser.add_argument('--schema_prompt_file', '-spf', type=str,
                      default=None,
                      help='Path to schema extraction prompt file. '
                           'Defaults to annotation/prompts/schema_extraction_prompt.txt')
  parser.add_argument('--gemini_workers', '-gw', type=int, default=8,
                      help='Max parallel Gemini API workers for schema assignment')

  # Other
  parser.add_argument('--seed', '-s', type=int, default=42)

  args = parser.parse_args()

  # --- Validation ---
  if args.variation_rate < 1:
    raise ValueError(f"--variation_rate must be >= 1, got {args.variation_rate}")
  if args.num_iterations > 0 and args.variation_rate > 1 and args.real_data_path is None:
    raise ValueError("--real_data_path is required when num_iterations > 0 and variation_rate > 1")
  if args.evaluate and args.real_data_path is None:
    raise ValueError("--real_data_path is required when --evaluate is enabled")

  if args.num_iterations > 0 and args.variation_rate > 1:
    assert args.rho > 0, "Iterative filtering requires a positive rho budget"
    rho = args.rho / args.num_iterations
  else:
    rho = None

  if args.variation_rate > 1:
    logging.info(f"Variation rate: {args.variation_rate}")
    logging.info(f"  -> Generate per iteration: {args.n_gen * args.variation_rate} samples")
    logging.info(f"  -> Filter to: {args.n_gen} samples")

  # Default schema prompt file
  if args.schema_name and args.schema_prompt_file is None:
    args.schema_prompt_file = osp.join(
      repo_root, 'annotation', 'prompts', 'schema_extraction_prompt.txt'
    )

  # --- Seed ---
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  # --- Model & Tokenizer ---
  device = f'cuda:{args.device}'
  compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

  logging.info(f"Loading pretrained model: {args.model_name_or_path}")
  model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=compute_dtype,
    low_cpu_mem_usage=True,
    attn_implementation='sdpa',
    token=hf_token,
  )
  model.eval()
  model.to(device)

  # Compile model for additional speedup (PyTorch 2.0+)
  if hasattr(torch, 'compile'):
    logging.info("Compiling model with torch.compile()...")
    model = torch.compile(model, mode='reduce-overhead')

  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=hf_token)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
  tokenizer.padding_side = 'left'

  # --- Output Setup ---
  os.makedirs(args.output_dir, exist_ok=True)
  output_file = osp.join(args.output_dir, args.out_file)

  # --- Initial input data: n_gen empty items (no schema feature) ---
  input_data = [{} for _ in range(args.n_gen)]
  logging.info(f"Preparing {len(input_data)} samples for generation (no schema input)...")

  # --- Terminators ---
  terminators = [tokenizer.eos_token_id]
  for t in ["<|im_end|>", "<|endoftext|>"]:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
      tid = tokenizer.convert_tokens_to_ids(t)
      if isinstance(tid, int):
        terminators.append(tid)

  # --- Track evaluation results ---
  all_eval_results = []

  # --- Generation Loop ---
  if args.num_iterations == 0:
    # ===== SINGLE GENERATION =====
    t_start = time.time()
    generated_data = run_generation(
      model, tokenizer, input_data, args, device, terminators, prompt_type='gen'
    )
    elapsed = time.time() - t_start
    logging.info(f"Generation time: {elapsed:.2f}s ({elapsed/60:.2f} min)")

    save_jsonl(generated_data, output_file)

    if args.evaluate:
      eval_results = run_evaluation(output_file, args.real_data_path, args.output_dir, 1, device, real_text_column=args.text_column)
      if eval_results:
        all_eval_results.append(eval_results)

  else:
    # ===== ITERATIVE GENERATION + FILTERING =====
    for iteration in range(1, args.num_iterations + 1):
      logging.info(f"\n{'='*60}")
      logging.info(f"Starting Iteration {iteration}/{args.num_iterations}")
      logging.info(f"{'='*60}\n")

      prompt_type = 'gen' if iteration == 1 else 'variation'

      if args.variation_rate > 1:
        expanded_input = [item for item in input_data for _ in range(args.variation_rate)]
        logging.info(
          f"Expanded {len(input_data)} -> {len(expanded_input)} samples "
          f"(variation_rate={args.variation_rate})"
        )
      else:
        expanded_input = input_data

      t_start = time.time()
      generated_data = run_generation(
        model, tokenizer, expanded_input, args, device, terminators, prompt_type
      )
      elapsed = time.time() - t_start
      logging.info(f"Generation time: {elapsed:.2f}s ({elapsed/60:.2f} min)")

      if args.variation_rate > 1:
        logging.info(f"Filtering {len(generated_data)} -> {args.n_gen} samples...")
        filtered_data = filter_by_voting(
          generated_data,
          args.real_data_path,
          args.n_gen,
          embedding_model_name=args.embedding_model,
          device=device,
          rho=rho,
          text_column=args.text_column,
        )
      else:
        filtered_data = generated_data

      save_jsonl(filtered_data, output_file)

      # Evaluate if requested
      if args.evaluate:
        eval_results = run_evaluation(output_file, args.real_data_path, args.output_dir, iteration, device, real_text_column=args.text_column)
        if eval_results:
          all_eval_results.append(eval_results)

      # Prepare input for next iteration
      input_data = filtered_data

  # --- Save evaluation summary ---
  if args.evaluate and all_eval_results:
    # Use generated data filename in the summary filename
    base_filename = os.path.splitext(args.out_file)[0]
    summary_file = os.path.join(args.output_dir, f'eval_summary_{base_filename}.json')
    with open(summary_file, 'w') as f:
      json.dump(all_eval_results, f, indent=2, ensure_ascii=False)
    logging.info(f"Evaluation summary saved to: {summary_file}")

  # --- Post-hoc Schema Assignment ---
  if args.schema_name:
    logging.info(f"\n{'='*60}")
    logging.info(f"Post-hoc schema assignment using schema: {args.schema_name}")
    logging.info(f"{'='*60}\n")

    # Reload final generated data (may have been saved by last iteration)
    final_data = load_jsonl(output_file)
    final_data = assign_schemas(
      final_data,
      schema_name=args.schema_name,
      schema_prompt_file=args.schema_prompt_file,
      max_workers=args.gemini_workers,
    )
    save_jsonl(final_data, output_file)
    logging.info("Schema assignment written back to output file.")
  else:
    logging.info("No --schema_name provided; skipping post-hoc schema assignment.")

  # --- Summary ---
  logging.info(f"\n{'='*60}")
  logging.info("Done!")
  logging.info(f"Final output: {output_file}")
  logging.info(f"{'='*60}")


if __name__ == '__main__':
  main()
