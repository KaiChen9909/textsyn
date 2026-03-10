"""Two-step text generation with voting-based filtering.

Steps:
1. Generate L * n_gen texts unconditionally (without schema/prompt, like generation_biorxiv_gen.py)
2. Filter to top n_gen texts via nearest-neighbor voting from real data (like generation_condgen_filter.py)
"""

# --- Standard Library Imports ---
import argparse
import json
import logging
import os
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
  """Main function to run the two-step generation and filtering process."""
  # --- Argument Parsing ---
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', '-d', type=int, default=0)
  parser.add_argument('--model_name_or_path', '-m', type=str, required=True)
  parser.add_argument('--output_dir', '-o', type=str, default='')
  parser.add_argument(
      '--out_file', '-out', type=str, default='output.json', required=True
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

  args = parser.parse_args()

  # --- Validation ---
  assert args.L >= 1, f"L must be >= 1, got {args.L}"

  # --- random seed ---
  seed = args.seed
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # --- Device Setup and Model Loading ---
  device = f'cuda:{args.device}'

  compute_dtype = (
      torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

  logging.info("Loading model...")

  model = AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      torch_dtype=compute_dtype,
      low_cpu_mem_usage=True,
      attn_implementation='eager',
      token=hf_token,
  )
  model.eval()
  model.to(device)

  tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    token=hf_token,
  )
  tokenizer.padding_side = 'left'

  # --- Prompt Preparation ---
  prompt_dict = get_prompt_dict(f'{args.prompt_str}_generation')
  prompt_text = prompt_dict['prompt']
  logging.info('----PROMPT----')
  logging.info(prompt_text)
  logging.info('---END PROMPT---')

  # Tokenize the prompt once
  input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

  # --- Output Directory Setup ---
  out_folder = (
      f'results/intermediate/generations_{args.output_dir}' if args.output_dir else 'generations'
  )
  os.makedirs(out_folder, exist_ok=True)

  n_gen = args.n_gen
  n_total = n_gen * args.L  # total number of texts to generate before filtering
  bs = args.bs
  jsonl_file = os.path.join(out_folder, args.out_file)

  # --- Step 1: Unconditional Generation ---
  t_start = time.time()
  all_generated = []

  logging.info(f'Step 1: Generating {n_total} sequences unconditionally...')

  for i in tqdm(range(0, n_total, bs)):
    bs_cur = min(bs, n_total - i)

    output = model.generate(
        input_ids,
        num_return_sequences=bs_cur,
        max_new_tokens=args.seq_len,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    output = output[:, input_ids.shape[1]:]

    output_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

    for sample in output_texts:
      all_generated.append({
          'input_text': '',  # No input text for unconditional generation
          'generated_text': sample,
      })

  t_gen = time.time() - t_start
  logging.info(f'Generation complete. Generated {len(all_generated)} samples in {t_gen:.2f}s')

  # --- Step 2: Filter by voting ---
  if args.L > 1:
    logging.info(f'Step 2: Filtering {len(all_generated)} -> {n_gen} samples via voting...')
    t_filter_start = time.time()

    rho = args.rho if args.rho > 0 else None
    filtered_data, vote_counts, selected_indices = filter_by_voting(
      all_generated,
      args.real_data_path,
      top_k=n_gen,
      embedding_model_name=args.embedding_model,
      device=device,
      rho=rho,
      text_column=args.text_column,
    )

    t_filter = time.time() - t_filter_start
    logging.info(f'Filtering complete in {t_filter:.2f}s')
  else:
    logging.info('L=1, skipping filtering step.')
    filtered_data = all_generated
    vote_counts = np.ones(len(all_generated))
    selected_indices = np.arange(len(all_generated))

  # --- Step 3: Save filtered results (unchanged) ---
  with open(jsonl_file, 'w') as f:
    for item in filtered_data:
      f.write(
          json.dumps(item, ensure_ascii=False) + '\n'
      )

  logging.info(f'Saved {len(filtered_data)} samples to {jsonl_file}')

  # --- Step 4: Save full dataset with labels ---
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
  logging.info(f'Total time: {time.time() - t_start:.2f}s')


if __name__ == '__main__':
  main()
