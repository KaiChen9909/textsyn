"""Analyze token distribution of OpenReview dataset using Gemma-3-1B tokenizer."""

import os
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv(".env")
hf_token = os.getenv("HF_LOGIN_STR")


def load_data(data_dir):
    """Load all CSV files from the data directory.

    Args:
        data_dir: Directory containing train.csv, validation.csv, test.csv

    Returns:
        Dictionary with split names as keys and DataFrames as values
    """
    splits = {}
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(data_dir, f'{split}.csv')
        if os.path.exists(file_path):
            splits[split] = pd.read_csv(file_path)
            print(f"Loaded {split}: {len(splits[split])} records")
        else:
            print(f"Warning: {file_path} not found")
    return splits


def compute_token_counts(texts, tokenizer):
    """Compute token counts for a list of texts.

    Args:
        texts: List of text strings
        tokenizer: Hugging Face tokenizer

    Returns:
        List of token counts
    """
    token_counts = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_counts.append(len(tokens))
    return token_counts


def print_statistics(token_counts, split_name):
    """Compute and print statistics for token counts.

    Args:
        token_counts: List of token counts
        split_name: Name of the data split
    """
    token_counts = np.array(token_counts)

    print(f"\n{'='*60}")
    print(f"Statistics for {split_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(token_counts)}")
    print(f"Mean: {token_counts.mean():.2f}")
    print(f"Median: {np.median(token_counts):.2f}")
    print(f"Std: {token_counts.std():.2f}")
    print(f"Min: {token_counts.min()}")
    print(f"Max: {token_counts.max()}")
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(token_counts, p):.2f}")

    # Print token count distribution bins
    print(f"\nToken count distribution:")
    bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, float('inf')]
    bin_labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-400', '400-500', '500+']

    for i in range(len(bins)-1):
        if bins[i+1] == float('inf'):
            count = np.sum(token_counts >= bins[i])
            percentage = (count / len(token_counts)) * 100
            print(f"  {bin_labels[i]}: {count} ({percentage:.2f}%)")
        else:
            count = np.sum((token_counts >= bins[i]) & (token_counts < bins[i+1]))
            percentage = (count / len(token_counts)) * 100
            print(f"  {bin_labels[i]}: {count} ({percentage:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze token distribution of OpenReview dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/openreview',
        help='Directory containing the dataset CSV files'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='google/gemma-3-1b-pt',
        help='Model name or path for tokenizer'
    )
    parser.add_argument(
        '--text_column',
        type=str,
        default='text',
        help='Name of the column containing text data'
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    print(f"Tokenizer loaded successfully")

    # Load data
    print(f"\nLoading data from: {args.data_dir}")
    splits_data = load_data(args.data_dir)

    if not splits_data:
        print("Error: No data files found!")
        return

    # Compute token counts for each split
    for split_name, df in splits_data.items():
        if args.text_column not in df.columns:
            print(f"Warning: Column '{args.text_column}' not found in {split_name}")
            print(f"Available columns: {df.columns.tolist()}")
            continue

        print(f"\nProcessing {split_name}...")
        texts = df[args.text_column].tolist()
        token_counts = compute_token_counts(texts, tokenizer)
        print_statistics(token_counts, split_name)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
