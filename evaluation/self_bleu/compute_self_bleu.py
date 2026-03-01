"""Compute Self-BLEU score for evaluating diversity of generated text.

Self-BLEU measures the diversity of a set of generated sentences by computing
the BLEU score of each sentence against all other sentences as references.
Lower Self-BLEU indicates higher diversity.

Reference:
  Zhu et al., "Texygen: A Benchmarking Platform for Text Generation Models"
  https://arxiv.org/abs/1802.01886
"""

import argparse
import json
import logging
import os
import sys
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def tokenize(text):
  """Simple whitespace tokenization."""
  if not isinstance(text, str):
    return []
  return text.strip().split()


def compute_bleu_for_sentence(args):
  """Compute BLEU score for a single sentence against all other sentences."""
  idx, sentences, n_gram, smoothing_function = args
  hypothesis = sentences[idx]
  references = [sentences[j] for j in range(len(sentences)) if j != idx]

  if not hypothesis or not references:
    return 0.0

  # Set weights based on n-gram
  if n_gram == 1:
    weights = (1.0, 0, 0, 0)
  elif n_gram == 2:
    weights = (0.5, 0.5, 0, 0)
  elif n_gram == 3:
    weights = (1/3, 1/3, 1/3, 0)
  elif n_gram == 4:
    weights = (0.25, 0.25, 0.25, 0.25)
  elif n_gram == 5:
    weights = (0.2, 0.2, 0.2, 0.2, 0.2)
  else:
    # Default to BLEU-4
    weights = (0.25, 0.25, 0.25, 0.25)

  try:
    score = sentence_bleu(
        references,
        hypothesis,
        weights=weights,
        smoothing_function=smoothing_function,
    )
  except Exception:
    score = 0.0

  return score


def compute_self_bleu(
    texts,
    n_gram=4,
    sample_size=None,
    num_workers=1,
    seed=42,
):
  """Compute Self-BLEU score for a list of texts.

  Args:
    texts: List of text strings.
    n_gram: The n-gram order for BLEU computation (1-5). Default is 4.
    sample_size: Number of sentences to sample for computation.
                 If None or >= len(texts), use all sentences.
    num_workers: Number of parallel workers. Default is 1.
    seed: Random seed for sampling. Default is 42.

  Returns:
    Dictionary containing self_bleu score and other statistics.
  """
  # Tokenize all texts
  sentences = [tokenize(text) for text in texts]

  # Filter out empty sentences
  sentences = [s for s in sentences if len(s) > 0]

  if len(sentences) < 2:
    logging.warning("Not enough valid sentences for Self-BLEU computation.")
    return {"self_bleu": 0.0, "num_sentences": len(sentences)}

  # Sample if needed
  if sample_size is not None and sample_size < len(sentences):
    np.random.seed(seed)
    indices = np.random.choice(len(sentences), size=sample_size, replace=False)
    sentences = [sentences[i] for i in indices]
    logging.info(f"Sampled {sample_size} sentences for Self-BLEU computation.")

  logging.info(f"Computing Self-BLEU-{n_gram} on {len(sentences)} sentences...")

  # Use smoothing to handle short sentences
  smoothing_function = SmoothingFunction().method1

  # Prepare arguments for parallel processing
  args_list = [
      (i, sentences, n_gram, smoothing_function)
      for i in range(len(sentences))
  ]

  # Compute BLEU scores
  if num_workers > 1:
    with Pool(num_workers) as pool:
      bleu_scores = list(
          tqdm(
              pool.imap(compute_bleu_for_sentence, args_list),
              total=len(sentences),
              desc=f"Self-BLEU-{n_gram}",
          )
      )
  else:
    bleu_scores = []
    for args in tqdm(args_list, desc=f"Self-BLEU-{n_gram}"):
      bleu_scores.append(compute_bleu_for_sentence(args))

  self_bleu = np.mean(bleu_scores)

  return {
      "self_bleu": float(self_bleu),
      "self_bleu_std": float(np.std(bleu_scores)),
      "num_sentences": len(sentences),
      "n_gram": n_gram,
  }


def main():
  # --- Argument Parsing ---
  parser = argparse.ArgumentParser(
      description="Compute Self-BLEU score for generated text diversity."
  )
  parser.add_argument(
      "--input_path",
      type=str,
      required=True,
      help="Path to the input CSV or JSONL file.",
  )
  parser.add_argument(
      "--text_column_name",
      type=str,
      default="abstract",
      help="Name of the column containing the text to evaluate.",
  )
  parser.add_argument(
      "--n_gram",
      type=int,
      default=4,
      choices=[1, 2, 3, 4, 5],
      help="N-gram order for BLEU computation (1-5). Default is 4.",
  )
  parser.add_argument(
      "--sample_size",
      type=int,
      default=None,
      help="Number of sentences to sample. Default is None (use all).",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=1,
      help="Number of parallel workers. Default is 1.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for sampling. Default is 42.",
  )
  parser.add_argument(
      "--save_path",
      type=str,
      default=None,
      help="Path to save results as JSON.",
  )
  args = parser.parse_args()

  # --- Load Data ---
  logging.info(f"Reading data from {args.input_path}...")
  try:
    if args.input_path.endswith(".jsonl"):
      df = pd.read_json(args.input_path, lines=True)
    elif args.input_path.endswith(".csv"):
      df = pd.read_csv(args.input_path)
    else:
      logging.error(
          "Unsupported file format. Please provide a CSV or JSONL file."
      )
      sys.exit(1)
    texts = df[args.text_column_name].tolist()
    logging.info(f"Found {len(texts)} texts to evaluate.")
  except (FileNotFoundError, KeyError, Exception) as e:
    logging.error(f"Error reading input file '{args.input_path}': {e}")
    sys.exit(1)

  # --- Compute Self-BLEU ---
  result = compute_self_bleu(
      texts=texts,
      n_gram=args.n_gram,
      sample_size=args.sample_size,
      num_workers=args.num_workers,
      seed=args.seed,
  )

  # --- Log Results ---
  logging.info(f"Self-BLEU-{args.n_gram}: {result['self_bleu']:.6f}")
  logging.info(f"Std: {result['self_bleu_std']:.6f}")
  logging.info(f"Num sentences: {result['num_sentences']}")

  # --- Save Results ---
  if args.save_path and args.save_path != "none":
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    result["input_path"] = args.input_path
    result["text_column_name"] = args.text_column_name
    with open(args.save_path, "w") as f:
      json.dump(result, f, indent=2)
    logging.info(f"Results saved to: {args.save_path}")

  logging.info("Done.")


if __name__ == "__main__":
  main()
