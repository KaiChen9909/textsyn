"""Fine-tunes a causal language model on synthetic data and evaluates
next-token prediction accuracy on real data.

This script trains a language model on synthetic text data and computes
the next-token prediction accuracy on real validation data.
"""

# --- Standard Library Imports ---
import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# --- Third-Party Library Imports ---
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    train_file: str = field(
        metadata={"help": "Path to the training data file (synthetic data). Can be CSV, JSON, or TXT."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the validation data file. If not provided, splits from training data."}
    )
    train_text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the training dataset containing the text data."},
    )
    validation_text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the validation dataset containing the text data. If not specified, uses train_text_column."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )


def main():
    # --- Parse arguments ---
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- Setup logging ---
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # --- Detecting last checkpoint ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # --- Set seed for reproducibility ---
    set_seed(training_args.seed)

    # --- Load datasets ---
    logger.info("Loading datasets...")
    data_files = {}
    dataset_args = {}

    def get_extension(path):
        ext = path.split(".")[-1]
        if ext == "jsonl":
            return "json"
        if ext == "txt":
            return "text"
        return ext

    from datasets import DatasetDict
    raw_datasets = DatasetDict()
    if data_args.train_file is not None:
        train_ext = get_extension(data_args.train_file)
        train_kwargs = {"keep_linebreaks": True} if train_ext == "text" else {}
        raw_datasets["train"] = load_dataset(
            train_ext, data_files={"train": data_args.train_file}, **train_kwargs
        )["train"]
    if data_args.validation_file is not None:
        val_ext = get_extension(data_args.validation_file)
        val_kwargs = {"keep_linebreaks": True} if val_ext == "text" else {}
        raw_datasets["validation"] = load_dataset(
            val_ext, data_files={"validation": data_args.validation_file}, **val_kwargs
        )["validation"]

    logger.info(f"Train dataset size: {len(raw_datasets['train'])}")
    if "validation" in raw_datasets:
        logger.info(f"Validation dataset size: {len(raw_datasets['validation'])}")

    # --- Load model configuration ---
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # Set padding token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )

    # We resize the embeddings only when necessary to avoid index errors
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # --- Preprocessing the datasets ---
    train_column_names = raw_datasets["train"].column_names
    train_text_column = data_args.train_text_column if data_args.train_text_column in train_column_names else train_column_names[0]
    logger.info(f"Using text column for training: {train_text_column}")

    def tokenize_function(text_column):
        def _tokenize(examples):
            return tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=data_args.max_seq_length,
            )
        return _tokenize

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_train = raw_datasets["train"].map(
            tokenize_function(train_text_column),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on training dataset",
        )

        tokenized_validation = None
        if "validation" in raw_datasets:
            val_column_names = raw_datasets["validation"].column_names
            val_text_column = data_args.validation_text_column if data_args.validation_text_column in val_column_names else val_column_names[0]
            logger.info(f"Using text column for validation: {val_text_column}")
            tokenized_validation = raw_datasets["validation"].map(
                tokenize_function(val_text_column),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=val_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if data_args.max_train_samples is not None:
        tokenized_train = tokenized_train.select(range(min(len(tokenized_train), data_args.max_train_samples)))

    if tokenized_validation is not None and data_args.max_eval_samples is not None:
        tokenized_validation = tokenized_validation.select(range(min(len(tokenized_validation), data_args.max_eval_samples)))

    train_dataset = tokenized_train
    eval_dataset = tokenized_validation

    # --- Data collator ---
    # DataCollatorForLanguageModeling automatically sets labels=input_ids,
    # masking padding tokens with -100 so they are excluded from the loss.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Metrics ---
    metric = evaluate.load("accuracy")

    def preprocess_logits_for_metrics(logits, labels):
        """Preprocess logits for metric computation."""
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        """Compute next-token prediction accuracy."""
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics

        # Shift predictions and labels for next-token prediction
        # predictions: shape (batch_size, seq_len)
        # We need to align: pred[i] should match label[i+1]
        preds = preds[:, :-1].reshape(-1)  # Remove last prediction
        labels = labels[:, 1:].reshape(-1)  # Remove first label (shift left)

        # Filter out padding tokens (marked as -100)
        mask = labels != -100
        preds = preds[mask]
        labels = labels[mask]

        accuracy = metric.compute(predictions=preds, references=labels)
        return accuracy

    # --- Initialize Trainer ---
    has_eval = eval_dataset is not None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if has_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if has_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if has_eval else None,
    )

    # --- Training ---
    if training_args.do_train:
        logger.info("*** Training on synthetic data ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    # --- Evaluation on real data ---
    if has_eval:
        logger.info("*** Evaluating on real data ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        # Calculate perplexity
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.json")
        with open(output_eval_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation results saved to {output_eval_file}")
        logger.info(f"Next-token prediction accuracy: {metrics['eval_accuracy']:.4f}")
        logger.info(f"Perplexity: {metrics['perplexity']:.4f}")

    # --- Prediction (optional) ---
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(eval_dataset, metric_key_prefix="predict")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["predict_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["predict_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        logger.info(f"Next-token prediction accuracy: {metrics['predict_accuracy']:.4f}")
        logger.info(f"Perplexity: {metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()
