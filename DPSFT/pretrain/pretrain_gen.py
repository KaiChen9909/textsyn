"""Generate text samples using pretrained Hugging Face models for synthetic data creation.

This script supports both single-iteration generation and iterative refinement:
- Single iteration (default): Generate from schemas using 'gen' prompt
- Multi-iteration: Generate → Filter by voting → Refine with 'variation' prompt
- Optional automatic evaluation using MAUVE and FID
"""

import argparse
import json
import logging
import os
import os.path as osp
import random
import time

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

from utils.pretrain_utils import get_pretrain_generation_prompt_dict
from dotenv import load_dotenv

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
)

# Load environment variables from repo root
load_dotenv(os.path.join(repo_root, '.env'))
hf_token = os.getenv("HF_LOGIN_STR")


def load_jsonl(file_path):
  """Load JSONL file into list of dicts."""
  data = []
  with open(file_path, 'r') as f:
    for line in f:
      data.append(json.loads(line.strip()))
  return data


def filter_by_voting(generated_data, real_data_path, top_k, 
                     embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', 
                     device='cuda', rho=None, text_column='abstract'):
  """
  Filter generated data by voting mechanism.
  Each real sample votes for its nearest generated sample based on semantic similarity.
  """
  from sentence_transformers import SentenceTransformer

  logging.info(f"Loading real data from: {real_data_path}")
  # Load real data
  if real_data_path.endswith('.jsonl'):
    data = load_jsonl(real_data_path)
    real_texts = [item[text_column] for item in data]
  elif real_data_path.endswith('.csv'):
    df = pd.read_csv(real_data_path)
    real_texts = df[text_column].tolist()
  else:
    raise ValueError(f"Unsupported file format: {real_data_path}")

  logging.info(f"Loaded {len(real_texts)} real samples")

  # Extract generated texts
  generated_texts = [item['generated_text'] for item in generated_data]
  logging.info(f"Computing embeddings for {len(generated_texts)} generated samples")

  # Load embedding model
  embedding_model = SentenceTransformer(embedding_model_name, device=device)

  # Determine optimal batch size based on available GPU memory
  # Larger batch size = faster encoding
  encode_batch_size = 256 if 'MiniLM' in embedding_model_name else 128

  # Compute embeddings with larger batch size
  logging.info(f"Computing real embeddings with batch_size={encode_batch_size}...")
  real_embeddings = embedding_model.encode(
    real_texts, batch_size=encode_batch_size, show_progress_bar=True,
    device=device, convert_to_numpy=True, normalize_embeddings=True  # Pre-normalize for cosine similarity
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

  # Filter data
  filtered_data = [generated_data[i] for i in top_k_indices]

  return filtered_data


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


def generate_batch(model, tokenizer, prompts, args, device, terminators):
  """Generate text for a batch of prompts."""
  # Tokenize
  batch = tokenizer(
    prompts,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=args.prompt_len,
    add_special_tokens=False
  ).to(device)

  input_len = batch['input_ids'].shape[1]

  # Generate
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

  # Decode (slice off input tokens)
  generated_tokens = outputs[:, input_len:]
  decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

  return decoded_texts


def run_generation(model, tokenizer, input_data, args, device, terminators,
          prompt_type='gen', system_prompt="You are a helpful assistant. Follow the instructions strictly."):
  """
  Run generation for one iteration.

  Args:
    input_data: list of dicts with 'input_text' (and optionally 'generated_text' for variation).
                Caller is responsible for expanding by variation_rate before calling.
    prompt_type: 'gen' for first iteration, 'variation' for refinement iterations
  """
  logging.info(f"Running generation with prompt type: {prompt_type}, {len(input_data)} samples")

  # Get prompt template
  prompt_dict = get_pretrain_generation_prompt_dict(f'{args.prompt_str}_generation', prompt_type)
  user_template = prompt_dict['prompt']

  all_results = []

  for i in tqdm(range(0, len(input_data), args.bs), desc="Generating"):
    batch_data = input_data[i : i + args.bs]

    # Prepare formatted prompts
    formatted_prompts = []
    for item in batch_data:
      if prompt_type == 'gen':
        messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_template.format(feature=item['input_text'])}
        ]
      else:  # variation
        messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_template.format(
            text=item['generated_text'],
            feature=item['input_text']
          )}
        ]

      formatted_prompts.append(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      )

    # Generate
    decoded_texts = generate_batch(model, tokenizer, formatted_prompts, args, device, terminators)

    for item, out in zip(batch_data, decoded_texts):
      all_results.append({
        'input_text': item['input_text'],
        'generated_text': out.strip()
      })

  logging.info(f"Generation complete. Generated {len(all_results)} samples.")
  return all_results


def save_jsonl(data, file_path):
  """Save list of dicts to JSONL file."""
  with open(file_path, 'w') as f:
    for item in data:
      f.write(json.dumps(item, ensure_ascii=False) + '\n')
  logging.info(f"Saved {len(data)} samples to: {file_path}")


def main():
  parser = argparse.ArgumentParser(description='Generate synthetic data using pretrained models')

  # Model arguments
  parser.add_argument('--device', '-d', type=int, default=0)
  parser.add_argument('--model_name_or_path', '-m', type=str, required=True,
            help='HuggingFace Instruct model name or path')
  parser.add_argument('--rho', '-rho', type=float, default=0.0,
            help='privacy budget')

  # Data arguments
  parser.add_argument('--output_dir', '-o', type=str, required=True,
            help='Output directory (data/pretrain)')
  parser.add_argument('--out_file', '-out', type=str, required=True,
            help='Output filename')
  parser.add_argument('--prompt_file', '-pf', type=str, required=True,
            help='CSV file containing schema prompts from ALGO (e.g., JSON file generated by AIM)')
  parser.add_argument('--prompt_str', '-ps', type=str, required=True,
            help='Prompt template string (e.g., biorxiv_condgen_pretrain)')

  # Generation arguments
  parser.add_argument('--prompt_len', '-pl', type=int, default=300,
            help='Max input length')
  parser.add_argument('--seq_len', '-sl', type=int, default=512,
            help='Max new tokens to generate')
  parser.add_argument('--bs', '-bs', type=int, default=64,
            help='Batch size')
  parser.add_argument('--n_gen', '-n_gen', type=int, default=10000,
            help='Number of samples to generate in first iteration')
  parser.add_argument('--top_p', '-tp', type=float, default=0.95)
  parser.add_argument('--top_k', '-tk', type=int, default=0,
            help='Top-k sampling (0 to disable, default: 0)')
  parser.add_argument('--temperature', '-temp', type=float, default=1.0)

  # Iterative refinement arguments (optional)
  parser.add_argument('--num_iterations', '-ni', type=int, default=0,
            help='Number of iterations (default: 0, single generation with no filtering)')
  parser.add_argument('--real_data_path', '-rd', type=str, default=None,
            help='Path to real data for filtering and evaluation')
  parser.add_argument('--variation_rate', '-vr', type=int, default=1,
            help='Variation rate (>=1): generate n_gen * variation_rate samples per iteration, then filter to n_gen. 1 = no filtering.')
  parser.add_argument('--embedding_model', '-em', type=str,
            default='sentence-transformers/all-mpnet-base-v2',
            help='Embedding model for filtering (default: all-mpnet-base-v2, fast alternative: all-MiniLM-L6-v2)')
  parser.add_argument('--text_column', '-tc', type=str, default='abstract',
            help='Name of text column in real data CSV (default: abstract)')
  parser.add_argument('--evaluate', action='store_true',
            help='Run evaluation after each iteration (requires real_data_path)')

  # Other arguments
  parser.add_argument('--seed', '-s', type=int, default=42)

  args = parser.parse_args()

  # Validation
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

  # --- Seed Setup ---
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  # --- Model & Tokenizer ---
  device = f'cuda:{args.device}'

  compute_dtype = (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

  logging.info(f"Loading pretrained model: {args.model_name_or_path}")
  model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=compute_dtype,
    low_cpu_mem_usage=True,
    attn_implementation='sdpa',  # Use SDPA for 2-3x speedup
    token=hf_token,
  )
  model.eval()
  model.to(device)

  # Compile model for additional speedup (PyTorch 2.0+)
  if hasattr(torch, 'compile'):
    logging.info("Compiling model with torch.compile()...")
    model = torch.compile(model, mode='reduce-overhead')

  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=hf_token)

  # Ensure pad_token exists (critical for batch inference)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

  tokenizer.padding_side = 'left'  # Left padding is required for generation

  # --- Output Setup ---
  os.makedirs(args.output_dir, exist_ok=True)
  output_file = os.path.join(args.output_dir, args.out_file)

  # --- Load initial prompts/schemas ---
  logging.info(f"Loading prompts from: {args.prompt_file}")
  input_prompts = pd.read_csv(args.prompt_file)['generated_text'].tolist()

  if len(input_prompts) > args.n_gen:
    random.seed(args.seed)
    input_prompts = random.sample(input_prompts, k=args.n_gen)

  # Prepare initial input data
  input_data = [{'input_text': text} for text in input_prompts]
  logging.info(f'Processing {len(input_data)} samples for generation...')

  # --- Setup terminators ---
  terminators = [tokenizer.eos_token_id]
  for t in ["<|im_end|>", "<|endoftext|>"]:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
      tid = tokenizer.convert_tokens_to_ids(t)
      if isinstance(tid, int):
        terminators.append(tid)

  # --- Track evaluation results ---
  all_eval_results = []

  if args.num_iterations == 0:
    # ========== SINGLE GENERATION (no filtering) ==========
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
    # ========== ITERATIVE GENERATION + FILTERING LOOP ==========
    for iteration in range(1, args.num_iterations + 1):
      logging.info(f"\n{'='*60}")
      logging.info(f"Starting Iteration {iteration}/{args.num_iterations}")
      logging.info(f"{'='*60}\n")

      prompt_type = 'gen' if iteration == 1 else 'variation'

      # Expand input_data by variation_rate
      if args.variation_rate > 1:
        expanded_input = [item for item in input_data for _ in range(args.variation_rate)]
        logging.info(f"Expanded {len(input_data)} samples to {len(expanded_input)} (variation_rate={args.variation_rate})")
      else:
        expanded_input = input_data

      # Run generation
      t_start = time.time()
      generated_data = run_generation(
        model, tokenizer, expanded_input, args, device, terminators, prompt_type
      )
      elapsed = time.time() - t_start
      logging.info(f"Generation time: {elapsed:.2f}s ({elapsed/60:.2f} min)")

      # Filter to n_gen
      if args.variation_rate > 1:
        logging.info(f"Filtering {len(generated_data)} -> {args.n_gen} samples...")
        filtered_data = filter_by_voting(
          generated_data,
          args.real_data_path,
          args.n_gen,
          embedding_model_name=args.embedding_model,
          device=device,
          rho=rho,
          text_column=args.text_column
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

  # --- Final summary ---
  logging.info(f"\n{'='*60}")
  if args.num_iterations > 0:
    logging.info("Iterative generation complete!")
  else:
    logging.info("Generation complete!")
  logging.info(f"Final output: {output_file}")
  logging.info(f"{'='*60}")


if __name__ == '__main__':
  main()
