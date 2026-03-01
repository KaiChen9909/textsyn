"""Evaluation utilities for iterative generation.

This module provides functions for computing MAUVE and FID scores
by integrating with the existing evaluation scripts.
"""

import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import evaluation code
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
mauve_dir = os.path.join(project_root, 'evaluation', 'mauve')
sys.path.insert(0, mauve_dir)
from compute_mauve import compute_mauve as compute_mauve_score


def load_texts_from_file(file_path, text_column='generated_text'):
  """Load texts from JSONL or CSV file."""
  if file_path.endswith('.jsonl'):
    texts = []
    with open(file_path, 'r') as f:
      for line in f:
        line = line.strip()
        if not line:  # Skip empty lines
          continue
        item = json.loads(line)
        texts.append(item.get(text_column, ''))
  elif file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
    texts = df[text_column].tolist()
  else:
    raise ValueError(f"Unsupported file format: {file_path}")

  # Clean texts
  texts = [str(text) if text is not None else "" for text in texts]
  return texts


def compute_specter2_embeddings(texts, batch_size=16, device='cuda:0', max_length=512):
  """
  Compute Specter2 embeddings for texts.

  Args:
    texts: List of text strings
    batch_size: Batch size for embedding computation
    device: Device to use
    max_length: Maximum sequence length

  Returns:
    numpy array of embeddings, shape (len(texts), embedding_dim)
  """
  from adapters import AutoAdapterModel
  from transformers import AutoTokenizer

  logging.info("Loading Specter2 model...")
  tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
  model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
  model.load_adapter(
    "allenai/specter2", source="hf", load_as="specter2", set_active=True
  )
  model.to(device)
  model.eval()

  logging.info(f"Computing embeddings for {len(texts)} texts...")
  all_embeddings = []

  with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
      batch_texts = texts[i : i + batch_size]

      inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=max_length,
      ).to(device)

      output = model(**inputs)
      cls_embedding = output.last_hidden_state[:, 0, :]
      all_embeddings.append(cls_embedding.cpu())

  final_embeddings = torch.cat(all_embeddings, dim=0).numpy()
  logging.info(f"Embeddings shape: {final_embeddings.shape}")

  return final_embeddings


def compute_mauve_for_files(generated_file, real_file, output_dir, iteration,
              gen_text_column='generated_text', real_text_column='abstract', device='cuda:0'):
  """
  Compute MAUVE score between generated and real texts.

  Args:
    generated_file: Path to generated text file
    real_file: Path to real text file
    output_dir: Directory to save results
    iteration: Current iteration number
    gen_text_column: Name of text column in generated file
    real_text_column: Name of text column in real file
    device: Device to use

  Returns:
    Dictionary with MAUVE results
  """
  logging.info("Computing MAUVE score...")

  # Load texts
  generated_texts = load_texts_from_file(generated_file, gen_text_column)

  logging.info(f"Generated: {len(generated_texts)} samples")

  # Compute generated embeddings
  gen_embeddings = compute_specter2_embeddings(generated_texts, device=device)

  # Setup embeddings directory and filenames
  embeddings_dir = os.path.join(output_dir, 'embeddings')
  os.makedirs(embeddings_dir, exist_ok=True)

  # Extract base filename from generated_file for consistent naming
  base_filename = os.path.splitext(os.path.basename(generated_file))[0]
  gen_emb_path = os.path.join(embeddings_dir, f'{base_filename}_iter{iteration}_embeddings.npy')
  real_emb_path = os.path.join(embeddings_dir, f'real_embeddings.npy')

  # Save generated embeddings
  np.save(gen_emb_path, gen_embeddings)

  # Load or compute real embeddings (cache for efficiency)
  if os.path.exists(real_emb_path):
    logging.info(f"Loading cached real embeddings from: {real_emb_path}")
    real_embeddings = np.load(real_emb_path)
  else:
    logging.info("Computing real embeddings (first time, will be cached)...")
    real_texts = load_texts_from_file(real_file, real_text_column)
    logging.info(f"Real: {len(real_texts)} samples")
    real_embeddings = compute_specter2_embeddings(real_texts, device=device)
    np.save(real_emb_path, real_embeddings)
    logging.info(f"Cached real embeddings to: {real_emb_path}")

  # Compute MAUVE
  result = compute_mauve_score(
    p_features=gen_embeddings,
    q_features=real_embeddings,
    num_buckets='auto'
  )

  mauve_results = {
    'mauve': float(result.mauve),
    'frontier_integral': float(result.frontier_integral),
    'num_buckets': int(result.num_buckets),
  }

  logging.info(f"MAUVE score: {mauve_results['mauve']:.4f}")

  return mauve_results


def compute_fid_for_files(generated_file, real_file, output_dir, iteration,
              gen_text_column='generated_text', real_text_column='abstract', device='cuda:0'):
  """
  Compute FID score between generated and real texts.

  TODO: Implement FID computation if needed.
  FID (Fréchet Inception Distance) is typically used for images,
  but can be adapted for text using embeddings.

  Args:
    generated_file: Path to generated text file
    real_file: Path to real text file
    output_dir: Directory to save results
    iteration: Current iteration number
    gen_text_column: Name of text column in generated file
    real_text_column: Name of text column in real file
    device: Device to use

  Returns:
    Dictionary with FID results
  """
  logging.warning("FID computation not implemented yet")
  return {'fid': None}


def evaluate_iteration(generated_file, real_file, output_dir, iteration,
            gen_text_column='generated_text', real_text_column='abstract', device='cuda:0'):
  """
  Run full evaluation for an iteration.

  Args:
    generated_file: Path to generated text file
    real_file: Path to real text file
    output_dir: Directory to save results
    iteration: Current iteration number
    gen_text_column: Name of text column in generated file
    real_text_column: Name of text column in real file
    device: Device to use

  Returns:
    Dictionary with all evaluation results
  """
  import time

  logging.info(f"\n{'='*50}")
  logging.info(f"Evaluating iteration {iteration}")
  logging.info(f"{'='*50}")

  results = {
    'iteration': iteration,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
  }

  # Compute MAUVE
  try:
    mauve_results = compute_mauve_for_files(
      generated_file, real_file, output_dir, iteration,
      gen_text_column=gen_text_column, real_text_column=real_text_column, device=device
    )
    results.update(mauve_results)
  except Exception as e:
    logging.error(f"MAUVE computation failed: {e}")
    results['mauve'] = None
    results['frontier_integral'] = None

  # Compute FID (placeholder)
  try:
    fid_results = compute_fid_for_files(
      generated_file, real_file, output_dir, iteration,
      gen_text_column=gen_text_column, real_text_column=real_text_column, device=device
    )
    results.update(fid_results)
  except Exception as e:
    logging.error(f"FID computation failed: {e}")
    results['fid'] = None

  # Log results (individual files not saved, see eval_summary.json for all results)
  logging.info(f"Iteration {iteration} evaluation complete")
  logging.info(f"Results: {json.dumps(results, indent=2)}")

  return results
