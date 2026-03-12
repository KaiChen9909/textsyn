"""Generates text samples from a Hugging Face model given a conditional prompt,
then filters the results using embedding-based voting against real training data.

Steps:
1. Generate L * n_gen texts using prompts (same as generation_biorxiv_condgen.py)
2. Filter to top n_gen texts via nearest-neighbor voting from real data
"""

# --- Standard Library Imports ---
import argparse
import json
import logging
import os
import os.path as osp
import random
import time

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
# --- Application-Specific Imports ---
from utils.data_utils import get_prompt_dict
from dotenv import load_dotenv

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

load_dotenv("../.env")
hf_token = os.getenv("HF_LOGIN_STR")


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


def generate_variations_with_model(model, tokenizer, input_data, args, device, terminators):
  """
  Generate variations using an instruction model.
  Uses tokenizer.apply_chat_template() for automatic model compatibility.

  Args:
    model: HuggingFace model
    tokenizer: HuggingFace tokenizer
    input_data: List of dicts with 'input_text' (schema) and 'generated_text' (to vary)
    args: Command line arguments
    device: Device string
    terminators: List of terminator token IDs

  Returns:
    List of dicts with 'input_text' and 'generated_text' (varied)
  """
  logging.info(f"Generating variations for {len(input_data)} samples")

  # Get instruction text based on prompt_str
  if 'biorxiv' in args.prompt_str:
    instruction = (
      'Please rephrase the given synthetic scientific abstract, '
      'ensuring it belongs to the JSON summary, in the style of a bioRxiv paper.'
    )
  else:
    raise ValueError(f"Unsupported prompt_str for variation: {args.prompt_str}")

  all_variations = []

  for i in tqdm(range(0, len(input_data), args.bs), desc="Generating variations"):
    batch_data = input_data[i : i + args.bs]

    # Prepare prompts using chat template (auto-adapts to model format)
    prompts = []
    for item in batch_data:
      user_message = (
        f"{instruction}\n\n"
        f"JSON summary: {item['input_text']}\n\n"
        f"Original abstract: {item['generated_text']}"
      )

      messages = [
        {"role": "system", "content": "You are a helpful assistant. Follow the instructions strictly."},
        {"role": "user", "content": user_message}
      ]

      # Use apply_chat_template to format correctly for each model
      prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
      )
      prompts.append(prompt)

    # Tokenize
    batch = tokenizer(
      prompts,
      return_tensors='pt',
      padding=True,
      truncation=True,
      max_length=args.prompt_len,
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

    # Decode (slice off input)
    generated_tokens = outputs[:, input_len:]
    decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for item, varied_text in zip(batch_data, decoded_texts):
      all_variations.append({
        'input_text': item['input_text'],
        'generated_text': varied_text.strip()
      })

  logging.info(f"Generated {len(all_variations)} variations")
  return all_variations


def load_variation_model(model_path, device, hf_token):
  """Load variation model and tokenizer."""
  compute_dtype = (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

  logging.info(f"Loading variation model: {model_path}")

  # Check if it's a local path (for variation models, usually from HF hub)
  is_local = os.path.exists(model_path)

  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=compute_dtype,
    low_cpu_mem_usage=True,
    attn_implementation='eager',
    token=hf_token,
    local_files_only=is_local,
  )
  model.eval()
  model.to(device)

  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    token=hf_token,
    local_files_only=is_local,
  )
  tokenizer.padding_side = 'left'

  # Ensure pad_token exists
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

  # Setup terminators
  terminators = [tokenizer.eos_token_id]
  for t in ["<|im_end|>", "<|endoftext|>", "<end_of_turn>"]:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
      tid = tokenizer.convert_tokens_to_ids(t)
      if isinstance(tid, int) and tid > 0:
        terminators.append(tid)

  return model, tokenizer, terminators


def run_evaluation(generated_file, real_data_path, output_dir, round_idx,
                   device='cuda:0', text_column='abstract'):
  """Run MAUVE evaluation for current round."""
  try:
    from pretrain.eval_func import evaluate_iteration

    logging.info(f"Running evaluation for round {round_idx}...")
    results = evaluate_iteration(
      generated_file,
      real_data_path,
      output_dir,
      round_idx,
      gen_text_column='generated_text',
      real_text_column=text_column,
      device=device
    )
    return results
  except ImportError as e:
    logging.error(f"Failed to import eval_func: {e}")
    logging.error("Make sure pretrain/eval_func.py exists")
    return None
  except Exception as e:
    logging.warning(f"Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    return None


def save_eval_summary(eval_results, output_dir, output_filename):
  """Save all evaluation results to a summary file."""
  base_filename = os.path.splitext(output_filename)[0]
  summary_file = os.path.join(output_dir, f'eval_summary_{base_filename}.json')

  with open(summary_file, 'w') as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)

  logging.info(f"\n{'='*60}")
  logging.info(f"Evaluation summary saved to: {summary_file}")
  logging.info(f"{'='*60}")
  logging.info("MAUVE Scores by Round:")
  for result in eval_results:
    mauve = result.get('mauve', 'N/A')
    if isinstance(mauve, float):
      logging.info(f"  Round {result['iteration']}: {mauve:.4f}")
    else:
      logging.info(f"  Round {result['iteration']}: {mauve}")
  logging.info(f"{'='*60}\n")


def filter_by_voting(generated_data, real_data_path, top_k,
                     embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                     device='cuda', rho=None, text_column='abstract'):
  """
  Filter generated data by voting mechanism.
  Each real sample votes for its nearest generated sample based on semantic similarity.
  Select the top_k generated samples with the most votes.
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

  encode_batch_size = 256 if 'MiniLM' in embedding_model_name else 128

  # Compute embeddings
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

    generated_embeddings = np.ascontiguousarray(generated_embeddings.astype(np.float32))
    real_embeddings = np.ascontiguousarray(real_embeddings.astype(np.float32))

    d = generated_embeddings.shape[1]

    # Use IndexFlatIP for inner product (cosine similarity since embeddings are normalized)
    index = faiss.IndexFlatIP(d)

    if device.startswith('cuda'):
      try:
        gpu_id = int(device.split(':')[-1]) if ':' in device else 0
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        logging.info(f"Using FAISS GPU index on {device}")
      except Exception as e:
        logging.warning(f"Failed to move FAISS to GPU: {e}. Using CPU.")

    index.add(generated_embeddings)

    similarities, nearest_indices = index.search(real_embeddings, k=1)
    nearest_indices = nearest_indices.flatten()

  except ImportError:
    logging.warning("FAISS not available, falling back to numpy implementation")
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

  filtered_data = [generated_data[i] for i in top_k_indices]

  return filtered_data, vote_counts, top_k_indices


def main():
  """Main function to run the text generation, filtering, and saving process."""
  # --- Argument Parsing ---
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', '-d', type=int, default=0)
  parser.add_argument('--model_name_or_path', '-m', type=str, required=True)
  parser.add_argument('--output_dir', '-o', type=str, default='')
  parser.add_argument(
      '--out_file', '-out', type=str, default='output.json', required=True
  )
  parser.add_argument(
      '--prompt_len',
      '-pl',
      type=int,
      default=128,
      help='max sequence length for prompt',
  )
  parser.add_argument(
      '--seq_len',
      '-sl',
      type=int,
      default=512,
      help='max new tokens for generation',
  )
  parser.add_argument(
      '--bs', '-bs', type=int, default=64, help='batch size for generation'
  )
  parser.add_argument(
      '--n_gen',
      '-n_gen',
      type=int,
      default=64,
      help='number of final output texts after filtering',
  )
  parser.add_argument(
      '--L',
      '-L',
      type=int,
      default=2,
      help='oversampling factor: generate L * n_gen texts, then filter to n_gen',
  )
  parser.add_argument('--prompt_file', '-pf', type=str, required=True)
  parser.add_argument(
      '--top_p', '-tp', type=float, default=1.0, help='top_p for sampling'
  )
  parser.add_argument(
      '--top_k', '-tk', type=int, default=0, help='top_k for sampling'
  )
  parser.add_argument(
      '--temperature',
      '-temp',
      type=float,
      default=1.0,
      help='temperature for sampling',
  )
  parser.add_argument(
      '--begin_idx',
      '-b',
      type=int,
      default=0,
      help='begin index for input data',
  )
  parser.add_argument(
    '--seed',
    '-s',
    type=int,
    default=42,
    help='random seed for generation'
  )
  parser.add_argument('--prompt_str', '-ps', type=str, default='biorxiv')

  # --- Filter-related arguments ---
  parser.add_argument('--real_data_path', '-rd', type=str, required=True,
            help='path to real training data (csv or jsonl) for voting filter')
  parser.add_argument('--embedding_model', '-em', type=str,
            default='sentence-transformers/all-MiniLM-L6-v2',
            help='embedding model for voting filter')
  parser.add_argument('--text_column', '-tc', type=str, default='abstract',
            help='name of text column in real data')
  parser.add_argument('--rho', '-rho', type=float, default=0.0,
            help='DP noise parameter for voting (0 = no noise)')

  # --- Multi-round arguments ---
  parser.add_argument('--round', '-r', type=int, default=1,
            help='number of filtering rounds (default: 1, keep current behavior)')
  parser.add_argument('--variation_model', '-vm', type=str, default='Qwen/Qwen2.5-7B-Instruct',
            help='model for variation generation (round >= 2, default: Qwen2.5-7B-Instruct)')
  parser.add_argument('--evaluate', action='store_true',
            help='run MAUVE evaluation after each round')

  args = parser.parse_args()

  # --- Validation ---
  assert args.L >= 1, f"L must be >= 1, got {args.L}"

  # --- random seed ---
  seed = args.seed
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # --- Device Setup ---
  device = f'cuda:{args.device}'

  # --- Output Setup ---
  out_folder = (
      f'results/intermediate/generations_{args.output_dir}' if args.output_dir else 'generations'
  )
  os.makedirs(out_folder, exist_ok=True)
  jsonl_file = os.path.join(out_folder, args.out_file)

  # --- Load Round 1 Model (trained condgen model) ---
  compute_dtype = (
      torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

  logging.info(f"Loading Round 1 model (trained condgen model): {args.model_name_or_path}")

  # Check if it's a local path
  is_local = os.path.exists(args.model_name_or_path) or args.model_name_or_path.startswith('results/')

  model = AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      torch_dtype=compute_dtype,
      low_cpu_mem_usage=True,
      attn_implementation='eager',
      token=hf_token,
      local_files_only=is_local,
  )
  model.eval()
  model.to(device)

  tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    token=hf_token,
    attn_implementation="sdpa",
    local_files_only=is_local,
  )
  tokenizer.padding_side = 'left'

  # --- Prompt Preparation (Round 1 only) ---
  prompt_dict = get_prompt_dict(f'{args.prompt_str}_generation')
  prompt = prompt_dict['prompt']
  logging.info('-----------ROUND 1 PROMPT-----------')
  logging.info(prompt)
  logging.info('-----END PROMPT-----------')

  # --- Input Data Preparation ---
  input_data_raw = pd.read_csv(args.prompt_file)['generated_text'].tolist()  # results of AIM

  n_gen = args.n_gen

  # Sample n_gen prompts
  random.seed(42)
  if len(input_data_raw) >= n_gen:
    input_texts_unique = random.sample(input_data_raw, k=n_gen)
    logging.info(f'Sampled {n_gen} unique prompts from {len(input_data_raw)} available prompts (without replacement)')
  else:
    input_texts_unique = random.choices(input_data_raw, k=n_gen)
    logging.info(f'Sampled {n_gen} prompts with replacement from {len(input_data_raw)} available prompts ({len(set(input_texts_unique))} unique)')

  # Prepare initial input_data for round 1
  input_data = [{'input_text': text} for text in input_texts_unique]

  # --- Evaluation results tracker ---
  all_eval_results = []

  # --- Variation model variables ---
  variation_model = None
  variation_tokenizer = None
  variation_terminators = None

  # ========== MULTI-ROUND LOOP ==========
  for round_idx in range(1, args.round + 1):
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting Round {round_idx}/{args.round}")
    logging.info(f"{'='*60}\n")

    # Calculate privacy budget for this round
    rho_current = (args.rho / args.round) if args.rho > 0 else None
    if rho_current:
      logging.info(f"Privacy budget for this round: rho = {rho_current:.6f}")

    # --- Switch to variation model if needed (after round 1) ---
    if round_idx == 2:
      logging.info("Round 2: Switching to variation model...")
      logging.info(f"Releasing Round 1 model to free GPU memory...")
      del model
      del tokenizer
      torch.cuda.empty_cache()
      import gc
      gc.collect()

      # Load variation model
      variation_model, variation_tokenizer, variation_terminators = load_variation_model(
        args.variation_model, device, hf_token
      )

    # --- Expand input by L (oversampling) ---
    if args.L > 1:
      expanded_input = [item for item in input_data for _ in range(args.L)]
      logging.info(f"Expanded {len(input_data)} -> {len(expanded_input)} samples (L={args.L})")
    else:
      expanded_input = input_data

    # --- Generate ---
    t_start = time.time()

    if round_idx == 1:
      # Round 1: Use trained condgen model with original logic
      logging.info(f'Generating {len(expanded_input)} sequences with trained condgen model...')

      # Expand input_texts for round 1
      input_texts = [item['input_text'] for item in expanded_input]
      input_texts = sorted(input_texts, key=lambda x: len(tokenizer.encode(x)))
      logging.info('Rearranged input_texts by increasing tokenized length')

      all_generated = []
      bs = args.bs

      for i in tqdm(range(0, len(input_texts), bs), desc="Generating (Round 1)"):
        bs_cur = min(bs, len(input_texts) - i)

        cur_input_texts = input_texts[i : i + bs_cur]
        cur_prompts = [prompt.format(feature=text) for text in cur_input_texts]
        batch = tokenizer(
          cur_prompts,
          return_tensors='pt',
          padding=True,
          truncation=True,
          max_length=args.prompt_len,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
          output = model.generate(
            **batch,
            max_new_tokens=args.seq_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
          )
        output = output[:, batch['input_ids'].shape[1]:]

        output_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

        for input_text, sample in zip(cur_input_texts, output_texts):
          all_generated.append({
            'input_text': input_text,
            'generated_text': sample,
          })

    else:
      # Round 2+: Use variation model
      all_generated = generate_variations_with_model(
        variation_model, variation_tokenizer, expanded_input,
        args, device, variation_terminators
      )

    t_gen = time.time() - t_start
    logging.info(f'Generation complete in {t_gen:.2f}s ({t_gen/60:.2f} min)')

    # --- Filter by voting ---
    if args.L > 1:
      logging.info(f'Filtering {len(all_generated)} -> {n_gen} samples via voting...')
      t_filter_start = time.time()

      filtered_data, vote_counts, selected_indices = filter_by_voting(
        all_generated,
        args.real_data_path,
        top_k=n_gen,
        embedding_model_name=args.embedding_model,
        device=device,
        rho=rho_current,
        text_column=args.text_column,
      )

      t_filter = time.time() - t_filter_start
      logging.info(f'Filtering complete in {t_filter:.2f}s')
    else:
      logging.info('L=1, skipping filtering step.')
      filtered_data = all_generated
      vote_counts = np.ones(len(all_generated))
      selected_indices = np.arange(len(all_generated))

    # --- Save filtered results (overwrite each round) ---
    save_jsonl(filtered_data, jsonl_file)

    # --- Save full dataset with labels (only for last round) ---
    if round_idx == args.round:
      selected_set = set(selected_indices.tolist())
      full_out_file = 'full_' + args.out_file
      full_jsonl_file = os.path.join(out_folder, full_out_file)

      with open(full_jsonl_file, 'w') as f:
        for i, item in enumerate(all_generated):
          record = {
            'input_text': item['input_text'],
            'generated_text': item['generated_text'],
            'label': 1 if i in selected_set else 0,
            'vote_count': float(vote_counts[i]),
          }
          f.write(json.dumps(record, ensure_ascii=False) + '\n')

      logging.info(f'Saved {len(all_generated)} full samples (with labels) to {full_jsonl_file}')

    # --- Evaluate ---
    if args.evaluate:
      eval_results = run_evaluation(
        jsonl_file, args.real_data_path, out_folder,
        round_idx, device, args.text_column
      )
      if eval_results:
        all_eval_results.append(eval_results)

    # --- Prepare input for next round ---
    input_data = filtered_data

  # --- Save evaluation summary ---
  if args.evaluate and all_eval_results:
    save_eval_summary(all_eval_results, out_folder, args.out_file)

  # --- Final summary ---
  logging.info(f"\n{'='*60}")
  logging.info(f"Multi-round generation complete!")
  logging.info(f"Total rounds: {args.round}")
  logging.info(f"Final output: {jsonl_file}")
  logging.info(f"{'='*60}\n")


if __name__ == '__main__':
  main()
