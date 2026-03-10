"""Sample topics from topic distribution for text generation.

This script loads a topic distribution (saved from extract_topics.py)
and samples topics according to their frequency. For each sampled topic,
it retrieves the corresponding keywords to use as generation conditions.
"""

import argparse
import json
import logging
import os
import os.path as osp
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def load_topic_distribution(distribution_path: str) -> Dict:
    """Load topic distribution from JSON file.

    Args:
        distribution_path: Path to topic distribution JSON file

    Returns:
        Dictionary containing topic distribution
    """
    logging.info(f"Loading topic distribution from: {distribution_path}")

    with open(distribution_path, 'r') as f:
        distribution = json.load(f)

    logging.info(f"  Total topics: {distribution['n_topics']}")
    logging.info(f"  Total documents in original dataset: {distribution['total_documents']}")

    return distribution


def sample_topics(
    distribution: Dict,
    n_samples: int,
    rho: float,
    exclude_outliers: bool = True,
    seed: int = 42
) -> List[Tuple[int, str, List[str]]]:
    """Sample topics from the distribution.

    Args:
        distribution: Topic distribution dictionary
        n_samples: Number of samples to generate
        rho: DP budget
        sampling_mode: How to sample topics
            - 'frequency': Sample proportional to topic frequency
            - 'uniform': Sample uniformly across all topics
        exclude_outliers: Whether to exclude outlier topic (-1)
        seed: Random seed for reproducibility

    Returns:
        List of (topic_id, keywords_str, keywords_list) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    # Get topics and their frequencies
    topics_data = distribution['topics']

    # Filter outliers if requested
    if exclude_outliers:
        topics_data = {k: v for k, v in topics_data.items() if int(k) != -1}

    topic_ids = [int(k) for k in topics_data.keys()]
    keywords_map = {int(k): v['keywords'] for k, v in topics_data.items()}

    # Sample proportional to frequency with DP noise
    frequencies = [topics_data[str(tid)]['frequency'] for tid in topic_ids]
    frequencies = np.array(frequencies)

    # Add Gaussian noise for differential privacy
    if rho > 0:
        noise = np.sqrt(1/(2*rho)) * np.random.randn(*frequencies.shape)
        frequencies += noise
        frequencies = np.clip(frequencies, 0, None)

    # Normalize to valid probability distribution
    frequencies = frequencies / frequencies.sum()

    sampled_topic_ids = np.random.choice(
        topic_ids,
        size=n_samples,
        replace=True,
        p=frequencies
    )

    # Prepare output: (topic_id, keywords_str, keywords_list)
    samples = []
    for tid in sampled_topic_ids:
        keywords_list = keywords_map[tid]
        keywords_str = ", ".join(keywords_list)
        samples.append((int(tid), keywords_str, keywords_list))

    return samples


def save_samples(
    samples: List[Tuple[int, str, List[str]]],
    output_path: str,
    format: str = 'csv'
):
    """Save sampled topics to file.

    Args:
        samples: List of (topic_id, keywords_str, keywords_list) tuples
        output_path: Path to output file
        format: Output format ('csv' or 'jsonl')
    """
    logging.info(f"Saving {len(samples)} samples to: {output_path}")

    os.makedirs(osp.dirname(output_path), exist_ok=True)

    if format == 'csv':
        df = pd.DataFrame({
            'topic': [s[0] for s in samples],
            'schema': [s[1] for s in samples]
        })
        df.to_csv(output_path, index=False)

    elif format == 'jsonl':
        with open(output_path, 'w') as f:
            for topic_id, keywords_str, keywords_list in samples:
                record = {
                    'topic': topic_id,
                    'schema': keywords_str,
                    'keywords': keywords_list
                }
                f.write(json.dumps(record) + '\n')

    else:
        raise ValueError(f"Unknown format: {format}")

    logging.info(f"Samples saved successfully!")


def print_statistics(samples: List[Tuple[int, str, List[str]]]):
    """Print statistics about sampled topics.

    Args:
        samples: List of sampled topics
    """
    topic_ids = [s[0] for s in samples]
    unique_topics = set(topic_ids)

    logging.info(f"\nSampling Statistics:")
    logging.info(f"  Total samples: {len(samples)}")
    logging.info(f"  Unique topics: {len(unique_topics)}")
    logging.info(f"  Topic coverage: {len(unique_topics)} topics")

    # Top 5 most sampled topics
    from collections import Counter
    topic_counts = Counter(topic_ids)
    top_5 = topic_counts.most_common(5)

    logging.info(f"\n  Top 5 most sampled topics:")
    for topic_id, count in top_5:
        keywords = [s[1] for s in samples if s[0] == topic_id][0]
        logging.info(f"    Topic {topic_id}: {count} samples")
        logging.info(f"      Keywords: {keywords[:80]}...")


def main():
    """Main function to sample topics."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Sample topics from topic distribution"
    )
    parser.add_argument(
        '--distribution_path', '-d',
        type=str,
        required=True,
        help='Path to topic distribution JSON file'
    )
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        required=True,
        help='Path to output file'
    )
    parser.add_argument(
        '--n_samples', '-n',
        type=int,
        default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--rho', '-rho',
        type=float,
        default=0.0,
        help='zCDP privacy budget'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='csv',
        choices=['csv', 'jsonl'],
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--include_outliers',
        action='store_true',
        help='Include outlier topic (-1) in sampling (default: exclude)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # --- Load Distribution ---
    distribution = load_topic_distribution(args.distribution_path)

    # --- Sample Topics ---
    logging.info(f"\nSampling {args.n_samples} topics using '{args.sampling_mode}' mode...")
    samples = sample_topics(
        distribution=distribution,
        n_samples=args.n_samples,
        rho=args.rho,
        exclude_outliers=not args.include_outliers,
        seed=args.seed
    )

    # --- Print Statistics ---
    print_statistics(samples)

    # --- Save Samples ---
    save_samples(samples, args.output_file, format=args.format)

    # --- Print Examples ---
    logging.info(f"\nExample samples (first 5):")
    for i, (topic_id, keywords_str, keywords_list) in enumerate(samples[:5], 1):
        logging.info(f"\nSample {i}:")
        logging.info(f"  Topic ID: {topic_id}")
        logging.info(f"  Keywords: {keywords_str}")


if __name__ == '__main__':
    main()
