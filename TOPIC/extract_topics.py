"""Extract topic keywords from documents using BERTopic.

This script uses the pretrained CTCL-Topic model (trained on Wikipedia)
to extract topic keywords for each document in the dataset.

References:
    CTCL paper: https://github.com/tanyuqian/synthetic-private-data
"""

import argparse
import json
import logging
import os
import os.path as osp
from collections import Counter
from typing import List, Dict, Any, Tuple

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def load_bertopic_model(model_path: str, embedding_model: str = "all-MiniLM-L6-v2") -> BERTopic:
    """Load pretrained BERTopic model.

    Args:
        model_path: Path to the pretrained BERTopic model directory
        embedding_model: Name of the sentence transformer model (default: all-MiniLM-L6-v2)

    Returns:
        Loaded BERTopic model
    """
    logging.info(f"Loading BERTopic model from: {model_path}")
    logging.info(f"Using embedding model: {embedding_model}")

    if not osp.exists(model_path):
        raise FileNotFoundError(
            f"Model path {model_path} not found. "
            f"Please download the pretrained CTCL-Topic model first:\n"
            f"  bash download_model.sh"
        )

    # Load embedding model
    sentence_model = SentenceTransformer(embedding_model)

    # Load BERTopic model
    topic_model = BERTopic.load(model_path, embedding_model=sentence_model)

    # Get model info
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
    logging.info(f"Model loaded successfully!")
    logging.info(f"  Number of topics: {n_topics}")

    return topic_model


def extract_topics_and_keywords(
    topic_model: BERTopic,
    documents: List[str]
) -> Tuple[List[int], List[str]]:
    """Extract topic IDs and keywords for each document.

    Args:
        topic_model: Pretrained BERTopic model
        documents: List of documents to extract topics for

    Returns:
        Tuple of (topic_ids, keywords_list)
    """
    logging.info("Extracting topics for documents...")

    # Transform documents to get topic assignments
    topics, _ = topic_model.transform(documents)

    # Extract keywords for each document
    keywords_list = []
    for topic_id in tqdm(topics, desc="Extracting keywords"):
        if topic_id == -1:
            # Outlier topic, use empty keywords
            keywords_list.append("")
        else:
            # Get top 10 keywords for this topic
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                # Format as comma-separated list
                keywords = ", ".join([word for word, _ in topic_words[:10]])
                keywords_list.append(keywords)
            else:
                keywords_list.append("")

    return topics, keywords_list


def save_topic_distribution(
    topics: List[int],
    topic_model: BERTopic,
    output_path: str
):
    """Save topic frequency distribution.

    Args:
        topics: List of topic IDs
        topic_model: BERTopic model
        output_path: Path to save the distribution JSON file
    """
    logging.info("Computing topic distribution...")

    # Count topic frequencies
    topic_counter = Counter(topics)

    # Get topic information
    topic_info = topic_model.get_topic_info()

    # Build distribution with topic metadata
    distribution = {
        "total_documents": len(topics),
        "n_topics": len(topic_info) - 1,  # Exclude outlier topic
        "topics": {}
    }

    for topic_id, count in topic_counter.items():
        if topic_id == -1:
            topic_name = "Outlier"
            keywords = []
        else:
            topic_words = topic_model.get_topic(topic_id)
            keywords = [word for word, _ in topic_words[:10]]
            topic_name = "_".join(keywords[:3])  # First 3 keywords as name

        distribution["topics"][str(topic_id)] = {
            "count": int(count),
            "frequency": count / len(topics),
            "keywords": keywords,
            "name": topic_name
        }

    # Save to JSON
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(distribution, f, indent=2)

    logging.info(f"Saved topic distribution to: {output_path}")
    logging.info(f"  Total topics: {distribution['n_topics']}")
    logging.info(f"  Total documents: {distribution['total_documents']}")


def main():
    """Main function to extract topics and save results."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Extract topic keywords using pretrained CTCL-Topic model"
    )
    parser.add_argument(
        '--input_file', '-i',
        type=str,
        required=True,
        help='Path to input CSV file containing documents'
    )
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        required=True,
        help='Path to output CSV file with topic keywords'
    )
    parser.add_argument(
        '--dataset_name', '-dn',
        type=str,
        default='biorxiv',
        help='Name of the dataset (default: biorxiv)'
    )
    parser.add_argument(
        '--text_column', '-tc',
        type=str,
        default='abstract',
        help='Name of the column containing text (default: abstract)'
    )
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        required=True,
        help='Path to pretrained CTCL-Topic model (download with download_model.sh)'
    )
    parser.add_argument(
        '--distribution_path', '-dp',
        type=str,
        default=None,
        help='Path to save topic distribution (default: ./distribution/{dataset_name}_topic_distribution.json)'
    )
    parser.add_argument(
        '--embedding_model', '-em',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--keep_empty',
        action='store_true',
        help='Keep documents without topics (default: filter them out)'
    )

    args = parser.parse_args()

    # --- Load Data ---
    logging.info(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)

    if args.text_column not in df.columns:
        raise ValueError(
            f"Column '{args.text_column}' not found in input file. "
            f"Available columns: {df.columns.tolist()}"
        )

    documents = df[args.text_column].astype(str).tolist()
    logging.info(f"Loaded {len(documents)} documents")

    # --- Load Pretrained Model ---
    topic_model = load_bertopic_model(args.model_path, args.embedding_model)

    # --- Extract Topics and Keywords ---
    topic_ids, keywords = extract_topics_and_keywords(topic_model, documents)

    # --- Prepare Output DataFrame ---
    output_df = df.copy()
    output_df['topic'] = topic_ids
    output_df['schema'] = keywords

    # Filter out empty topics if requested
    if not args.keep_empty:
        n_before = len(output_df)
        output_df = output_df[
            (output_df['schema'].notna()) &
            (output_df['schema'] != '') &
            (output_df['topic'] != -1)
        ]
        n_after = len(output_df)
        if n_before > n_after:
            logging.info(f"Filtered out {n_before - n_after} documents with empty topics")

    # Keep only three columns: topic, schema, text_column
    cols = ['topic', 'schema', args.text_column]
    output_df = output_df[cols]

    # --- Save Results ---
    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
    logging.info(f"Saving results to: {args.output_file}")
    output_df.to_csv(args.output_file, index=False)

    # Print statistics
    logging.info(f"\nExtraction complete!")
    logging.info(f"  Total documents: {len(output_df)}")
    logging.info(f"  Unique topics: {output_df['topic'].nunique()}")
    logging.info(f"  Output columns: {output_df.columns.tolist()}")

    # --- Save Topic Distribution (Optional) ---
    if args.distribution_path is not None:
        # Save distribution when path is provided
        save_topic_distribution(topic_ids, topic_model, args.distribution_path)
        logging.info(f"Topic distribution saved.")
    else:
        logging.info("Skipping topic distribution (no path provided).")

    # Print example outputs
    logging.info("\nExample outputs (first 3 documents):")
    for i, row in output_df.head(3).iterrows():
        logging.info(f"\nDocument {i+1}:")
        logging.info(f"  Topic: {row['topic']}")
        logging.info(f"  Schema (Keywords): {row['schema'][:80]}...")
        logging.info(f"  Text: {str(row[args.text_column])[:80]}...")


if __name__ == '__main__':
    main()
