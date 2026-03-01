# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain a LoRA model on feature data (non-DP)."""

import argparse
import json
import logging
import math
import os
import sys

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import datasets
import numpy as np
import peft
from peft import get_peft_model
from peft import LoraConfig
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import get_scheduler
from transformers import SchedulerType
import wandb
from dotenv import load_dotenv

# Load .env file for HuggingFace token
# Use absolute path based on script location (textsyn/.env)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, '..', '..', '.env')
load_dotenv(_env_path)

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import general_utils

logger = get_logger(__name__)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain LLMs with LoRA on feature data (non-DP)."
    )
    # data arguments
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the training data file (csv or jsonl).",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="Path to the validation data file. If not provided, will skip validation.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the column containing the text data.",
    )
    # model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=-1,
        help="Total number of training steps to perform. Overrides num_train_epochs if > 0.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store the model and log.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--log_freq", type=int, default=100, help="Freq of loss logging."
    )
    parser.add_argument(
        "--access_token", type=str, default=None, help="Huggingface access token"
    )
    parser.add_argument(
        "--gradient_ckpt",
        action="store_true",
        help="Use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--no_eval_at_start",
        action="store_true",
        help="Do not do evaluation before training.",
    )
    # LoRA hyperparameters
    parser.add_argument(
        "--lora_r", type=int, default=8, help="Rank of LoRA fine-tuning."
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=16, help="Value of alpha for lora."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Dropout for LoRA."
    )
    # wandb
    parser.add_argument(
        "--wandb_project", type=str, default="pretrain-feature", help="Wandb project name."
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb logging."
    )

    args = parser.parse_args()
    return args


def load_data(data_path, text_column):
    """Load data from csv or jsonl file."""
    if data_path.endswith('.csv'):
        dataset = datasets.load_dataset('csv', data_files={'data': data_path})['data']
    elif data_path.endswith('.jsonl') or data_path.endswith('.json'):
        dataset = datasets.load_dataset('json', data_files={'data': data_path})['data']
    else:
        raise ValueError(f"Unsupported data format: {data_path}. Use csv or jsonl.")

    # Verify text column exists
    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found in data. Available columns: {dataset.column_names}")

    return dataset


def tokenize_function(examples, tokenizer, max_seq_length, text_column):
    """Tokenize the text for causal language modeling."""
    texts = examples[text_column]

    tokenized = tokenizer(
        texts,
        max_length=max_seq_length,
        padding=False,
        truncation=True,
    )

    # For CLM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()

    return tokenized


class DataCollatorForCLM:
    """Data collator for causal language modeling."""

    IGNORE_INDEX = -100

    def __init__(self, tokenizer, max_length=512, device='cuda'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, instances):
        input_ids = [torch.tensor(inst['input_ids']).long() for inst in instances]
        labels = [torch.tensor(inst['labels']).long() for inst in instances]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.IGNORE_INDEX
        )

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


def eval_epoch(model, accelerator, eval_dataloader, epoch, args, description='end'):
    """Evaluate model on eval dataset."""
    model.eval()
    eval_total_loss = torch.zeros(1, device=accelerator.device)
    eval_total_num_tokens = 0
    loss_denom = args.max_seq_length

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            with accelerator.no_sync(model):
                outputs = model(**batch, use_cache=False)
                labels = batch['labels']
                num_tokens = torch.sum(labels != -100).detach()
                eval_total_num_tokens += num_tokens
                loss = outputs.loss * (num_tokens / loss_denom)
                eval_total_loss += loss.detach()

                if step % 100 == 0:
                    progress = step * 100 / len(eval_dataloader)
                    accelerator.print(f'epoch {epoch}, eval progress {progress:.2f}%')

    eval_epoch_loss = (eval_total_loss * loss_denom) / eval_total_num_tokens
    eval_epoch_loss = torch.mean(accelerator.gather(eval_epoch_loss)).item()
    accelerator.print(
        f'At epoch {epoch} {description}, eval loss {eval_epoch_loss:.4f}, '
        f'perplexity {np.exp(eval_epoch_loss):.4f}'
    )

    return eval_epoch_loss


def train_epoch(
    model,
    tokenizer,
    accelerator,
    optimizer,
    lr_scheduler,
    train_dataloader,
    epoch,
    args,
    completed_steps,
    progress_bar,
):
    """Train model for one epoch."""
    model.train()

    recent_loss = torch.zeros(1, device=accelerator.device)
    total_loss = torch.zeros(1, device=accelerator.device)
    recent_num_tokens = 0
    total_num_tokens = 0
    loss_denom = args.max_seq_length

    accumulated_steps = 0
    for step, batch in enumerate(train_dataloader):
        with accelerator.no_sync(model):
            num_tokens = torch.sum(batch['labels'] != -100).detach()
            recent_num_tokens += num_tokens
            total_num_tokens += num_tokens

            outputs = model(**batch)
            loss = outputs.loss * (num_tokens / loss_denom)

            if accelerator.is_main_process and loss != loss:
                accelerator.print('loss is NaN, exiting...')
                sys.exit(-1)

            accelerator.backward(loss)

            recent_loss += loss.detach()
            total_loss += loss.detach()

            accumulated_steps += 1
            if (
                accumulated_steps == args.gradient_accumulation_steps
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                performed_optimizer_step = True
                accumulated_steps = 0
            else:
                performed_optimizer_step = False

            if performed_optimizer_step:
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % args.log_freq == 0:
                    current_loss = (recent_loss / recent_num_tokens) * loss_denom
                    accelerator.print(
                        f'epoch {epoch}, step {completed_steps}, train loss {current_loss.item():.4f}, '
                        f'lr {lr_scheduler.get_last_lr()[0]:.2e}'
                    )
                    recent_loss = torch.zeros(1, device=accelerator.device)
                    recent_num_tokens = 0

                if completed_steps >= args.max_train_steps:
                    break
                if len(train_dataloader) - 1 - step < args.gradient_accumulation_steps:
                    break

    epoch_loss = (total_loss / total_num_tokens) * loss_denom
    epoch_loss = torch.mean(accelerator.gather(epoch_loss)).item()

    return epoch_loss, completed_steps


def main():
    args = parse_args()

    # Setup accelerator FIRST before any CUDA operations
    accelerator = Accelerator(project_dir=args.output_dir)
    args.logical_batch_size = (
        accelerator.num_processes
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    is_main_process = accelerator.is_main_process

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Login to huggingface - use access_token arg or HF_LOGIN_STR from .env
    access_token = args.access_token or os.environ.get('HF_LOGIN_STR')
    if access_token:
        os.system("huggingface-cli login --token " + access_token)

    # Use bfloat16 if supported (check after accelerator init)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False
    )

    # Add pad token if not present
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    if args.gradient_ckpt:
        model.gradient_checkpointing_enable()

    # Setup LoRA
    if "gpt2" not in args.model_name_or_path:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["c_attn", "c_proj"],
        )

    # Check for checkpoint
    checkpoint_epoch = general_utils.find_newest_checkpoint_epoch(args.output_dir)
    if checkpoint_epoch >= 0:
        model = peft.PeftModel.from_pretrained(
            model, args.output_dir + f"/peftmodel_epoch{checkpoint_epoch}"
        )
        for p in model.named_parameters():
            if "lora_A" in p[0] or "lora_B" in p[0]:
                p[1].requires_grad = True
        logging.info(f"Continuing training from checkpoint epoch {checkpoint_epoch}")
    else:
        model = get_peft_model(model, lora_config)
        logging.info("No checkpoint found. Start training from scratch.")

    general_utils.print_trainable_parameters(model)

    # Setup wandb
    if is_main_process and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=args.output_dir.split("/")[-1],
        )

    model = accelerator.prepare(model)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load and tokenize data
    logging.info(f"Loading training data from {args.train_data_path}")
    train_dataset = load_data(args.train_data_path, args.text_column)

    logging.info(f"Tokenizing training data (max_seq_length={args.max_seq_length})")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length, args.text_column),
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names,
    )

    # Load validation data if provided
    eval_dataset = None
    if args.val_data_path:
        logging.info(f"Loading validation data from {args.val_data_path}")
        eval_dataset = load_data(args.val_data_path, args.text_column)
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer, args.max_seq_length, args.text_column),
            batched=True,
            num_proc=4,
            remove_columns=eval_dataset.column_names,
        )

    # Create data collator
    data_collator = DataCollatorForCLM(
        tokenizer,
        max_length=args.max_seq_length,
        device=accelerator.device,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        drop_last=True,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_batchsize = max(1, args.per_device_train_batch_size // 2)
        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=eval_batchsize,
        )

    # Setup optimizer (only LoRA parameters)
    require_grad_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        require_grad_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    # Calculate training steps
    num_update_steps_per_epoch = math.floor(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.num_train_steps > 0:
        args.max_train_steps = args.num_train_steps
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    save_epochs = np.unique(
        [int(args.num_train_epochs / 4 * (i + 1)) for i in range(3)]
        + [args.num_train_epochs - 1]
    )
    logging.info(f"save_epochs: {save_epochs}")

    # Setup lr scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Print training info
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info(f"Num examples = {len(train_dataset)}, Num Epochs = {args.num_train_epochs}")
    logger.info(f"Learning rate = {args.learning_rate}")
    logger.info(f"Batch size per device = {args.per_device_train_batch_size}, Total batch size = {total_batch_size}")
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}, Total optimization steps = {args.max_train_steps}")

    # Progress bar
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch >= 0 else 0

    # Load training states if resuming
    if checkpoint_epoch >= 0:
        completed_steps = general_utils.load_training_states(
            args.output_dir, optimizer, lr_scheduler, checkpoint_epoch
        )
        if accelerator.is_main_process:
            logging.info(f"Loaded training states from checkpoint epoch {checkpoint_epoch}")

    progress_bar.update(completed_steps)

    # Training loop
    init_eval_loss = -1
    for epoch in range(start_epoch, args.num_train_epochs):
        # Evaluation before training
        if epoch == 0 and not args.no_eval_at_start and eval_dataloader is not None:
            init_eval_loss = eval_epoch(
                model, accelerator, eval_dataloader, epoch, args, description="start"
            )
            if is_main_process and not args.no_wandb:
                wandb.log({"eval_loss": init_eval_loss})

        # Train
        epoch_loss, completed_steps = train_epoch(
            model,
            tokenizer,
            accelerator,
            optimizer,
            lr_scheduler,
            train_dataloader,
            epoch,
            args,
            completed_steps,
            progress_bar,
        )

        # Evaluate
        eval_epoch_loss = -1
        if eval_dataloader is not None:
            eval_epoch_loss = eval_epoch(
                model, accelerator, eval_dataloader, epoch, args, description="end"
            )

        if is_main_process and not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "eval_loss": eval_epoch_loss,
            })

        # Save checkpoint
        if accelerator.is_main_process and epoch in save_epochs:
            general_utils.save_checkpoint(
                args.output_dir,
                model.module,
                optimizer,
                lr_scheduler,
                completed_steps,
                epoch,
            )

            with open(os.path.join(args.output_dir, f"results_epoch{epoch}.json"), "w") as f:
                json.dump({
                    "train_loss": epoch_loss,
                    "eval_loss": eval_epoch_loss,
                    "init_eval_loss": init_eval_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                }, f)

            logger.info(f"Epoch {epoch} checkpoint saved at {args.output_dir}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
