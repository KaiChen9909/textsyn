"""Evaluate schema-based text generation quality using BERTScore.

This script reads a clean dataset with schema and text columns,
uses an open model to generate text based on schema, and computes
BERTScore between generated and original text.
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def load_model_and_tokenizer(model_name_or_path, device):
    """Load model and tokenizer from HuggingFace.

    Args:
        model_name_or_path: HuggingFace model name or path.
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    logging.info(f"Loading model from {model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        attn_implementation='eager',
    )
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


def generate_texts(
    model,
    tokenizer,
    df,
    prompt_template,
    text_column="abstract",
    batch_size=8,
    max_prompt_length=512,
    max_gen_length=512,
    temperature=1.0,
    top_p=0.95,
    top_k=0,
    device=None,
):
    """Generate texts based on schema using the model.

    Args:
        model: HuggingFace causal language model.
        tokenizer: Tokenizer for the model.
        df: DataFrame with 'schema' column.
        prompt_template: Prompt template with {feature} placeholder.
        text_column: Name of the text column (for reference).
        batch_size: Batch size for generation.
        max_prompt_length: Max length for prompt tokens.
        max_gen_length: Max new tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        device: Device to run on.

    Returns:
        List of generated texts.
    """
    if device is None:
        device = next(model.parameters()).device

    # Build prompts from schema
    prompts = []
    for _, row in df.iterrows():
        schema = row['schema']
        if isinstance(schema, str):
            # If schema is already a JSON string, use it directly
            feature_str = schema
        else:
            feature_str = json.dumps(schema)
        prompt = prompt_template.format(feature=feature_str)
        prompts.append(prompt)

    generated_texts = []

    logging.info(f"Generating {len(prompts)} texts...")
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + batch_size]

            batch = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model.generate(
                **batch,
                max_new_tokens=max_gen_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # Remove prompt tokens from output
            output = output[:, batch['input_ids'].shape[1]:]
            output_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
            generated_texts.extend(output_texts)

    return generated_texts


def compute_bertscore(references, candidates, lang="en", batch_size=32, device=None):
    """Compute BERTScore between references and candidates.

    Args:
        references: List of reference texts.
        candidates: List of candidate (generated) texts.
        lang: Language for BERTScore.
        batch_size: Batch size for BERTScore computation.
        device: Device to run on.

    Returns:
        Dict with precision, recall, f1 scores (per sample and average).
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        logging.error("bert_score not installed. Install with: pip install bert-score")
        raise

    logging.info("Computing BERTScore...")

    P, R, F1 = bert_score(
        candidates,
        references,
        lang=lang,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )

    results = {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
        "precision_mean": P.mean().item(),
        "recall_mean": R.mean().item(),
        "f1_mean": F1.mean().item(),
        "precision_std": P.std().item(),
        "recall_std": R.std().item(),
        "f1_std": F1.std().item(),
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate schema-based text generation using BERTScore"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV file containing the dataset (must have 'schema' and text columns)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/gemma-3-1b-pt",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="abstract",
        help="Name of the text column in the dataset (default: abstract)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: use all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Max length for prompt tokens (default: 512)",
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling parameter (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results (optional)",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Custom prompt template with {feature} placeholder. If not specified, uses default biorxiv prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--bertscore_batch_size",
        type=int,
        default=32,
        help="Batch size for BERTScore computation (default: 32)",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Save generated texts to output file",
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    logging.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    logging.info(f"Loaded {len(df)} samples")

    # Verify required columns exist
    if 'schema' not in df.columns:
        raise ValueError("Dataset must have a 'schema' column")
    if args.text_column not in df.columns:
        raise ValueError(f"Dataset must have a '{args.text_column}' column")

    # Sample if requested
    if args.sample_size is not None and args.sample_size < len(df):
        logging.info(f"Sampling {args.sample_size} samples from dataset")
        df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)

    # Build prompt template
    if args.prompt_template is not None:
        prompt_template = args.prompt_template
    else:
        # Default biorxiv noexample prompt
        instruction = (
            "Please generate a synthetic scientific abstract that belongs to the"
            " below category, in the style of a bioRxiv paper."
        )
        prompt_template = f"<start_of_turn>user\n{instruction}\n\n{{feature}}\n<end_of_turn>\n<start_of_turn>model\n"

    logging.info("--- Prompt Template ---")
    logging.info(prompt_template)
    logging.info("--- End Prompt Template ---")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.device)

    # Generate texts
    t_start = time.time()
    generated_texts = generate_texts(
        model=model,
        tokenizer=tokenizer,
        df=df,
        prompt_template=prompt_template,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_gen_length=args.max_gen_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
    )
    gen_time = time.time() - t_start
    logging.info(f"Generation completed in {gen_time:.2f} seconds")

    # Get reference texts
    reference_texts = df[args.text_column].tolist()

    # Compute BERTScore
    t_start = time.time()
    bertscore_results = compute_bertscore(
        references=reference_texts,
        candidates=generated_texts,
        lang="en",
        batch_size=args.bertscore_batch_size,
        device=args.device,
    )
    score_time = time.time() - t_start
    logging.info(f"BERTScore computation completed in {score_time:.2f} seconds")

    # Print results
    logging.info("=" * 50)
    logging.info("BERTScore Results:")
    logging.info(f"  Precision: {bertscore_results['precision_mean']:.4f} (+/- {bertscore_results['precision_std']:.4f})")
    logging.info(f"  Recall:    {bertscore_results['recall_mean']:.4f} (+/- {bertscore_results['recall_std']:.4f})")
    logging.info(f"  F1:        {bertscore_results['f1_mean']:.4f} (+/- {bertscore_results['f1_std']:.4f})")
    logging.info("=" * 50)

    # Save results if output path specified
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)

        output_data = {
            "args": vars(args),
            "bertscore": {
                "precision_mean": bertscore_results['precision_mean'],
                "recall_mean": bertscore_results['recall_mean'],
                "f1_mean": bertscore_results['f1_mean'],
                "precision_std": bertscore_results['precision_std'],
                "recall_std": bertscore_results['recall_std'],
                "f1_std": bertscore_results['f1_std'],
            },
            "timing": {
                "generation_time": gen_time,
                "bertscore_time": score_time,
            },
            "num_samples": len(df),
        }

        if args.save_generations:
            output_data["per_sample"] = [
                {
                    "schema": row['schema'],
                    "reference": ref,
                    "generated": gen,
                    "precision": bertscore_results['precision'][i],
                    "recall": bertscore_results['recall'][i],
                    "f1": bertscore_results['f1'][i],
                }
                for i, (row, ref, gen) in enumerate(
                    zip(df.to_dict('records'), reference_texts, generated_texts)
                )
            ]

        with open(args.output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Results saved to {args.output_path}")

    return bertscore_results


if __name__ == "__main__":
    main()
