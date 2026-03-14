"""KTO (Kahneman-Tversky Optimization) training script.

Uses the full labeled dataset from gen_filter.sh (with label=1 for selected,
label=0 for rejected) to do preference-based RL training via trl's KTOTrainer.

Follows train_clm.py conventions:
- Saves LoRA adapters to output_dir/peftmodel_epoch{epoch}
- Merges and saves full models to output_dir/model_epoch{epoch}
"""

import argparse
import json
import logging
import os

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

from utils.data_utils import get_prompt_dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

load_dotenv('../.env')
hf_token = os.getenv('HF_LOGIN_STR')


def parse_args():
  parser = argparse.ArgumentParser(description='KTO RL training on labeled generation data.')

  # Model
  parser.add_argument('--model_name_or_path', type=str, required=True,
                       help='Path to the model to fine-tune (e.g. results/outputs/.../model_epoch79)')
  parser.add_argument('--access_token', type=str, default=None)

  # Data
  parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to full labeled dataset JSONL (from gen_filter.sh)')
  parser.add_argument('--prompt_style', type=str, default='biorxiv_condgen_filter_generation',
                       help='Prompt template name from get_prompt_dict()')

  # Output
  parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save the trained model')

  # Training hyperparameters
  parser.add_argument('--learning_rate', type=float, default=5e-6)
  parser.add_argument('--num_train_epochs', type=int, default=1)
  parser.add_argument('--max_steps', type=int, default=-1,
                       help='Max training steps. Overrides num_train_epochs if > 0.')
  parser.add_argument('--per_device_train_batch_size', type=int, default=4)
  parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
  parser.add_argument('--warmup_steps', type=int, default=10)
  parser.add_argument('--lr_scheduler_type', type=str, default='cosine')

  # KTO specific
  parser.add_argument('--beta', type=float, default=0.1,
                       help='KTO beta parameter (inverse temperature for KL penalty)')
  parser.add_argument('--max_prompt_length', type=int, default=300)
  parser.add_argument('--max_completion_length', type=int, default=512)

  # LoRA
  parser.add_argument('--lora_r', type=int, default=8)
  parser.add_argument('--lora_alpha', type=float, default=16)
  parser.add_argument('--lora_dropout', type=float, default=0.05)

  # Misc
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--logging_steps', type=int, default=5)
  parser.add_argument('--save_strategy', type=str, default='epoch')
  parser.add_argument('--gradient_checkpointing', action='store_true')
  parser.add_argument('--bf16', action='store_true', default=True)

  return parser.parse_args()


def load_labeled_dataset(dataset_path, prompt_template):
  """Load full labeled JSONL and format for KTO.

  KTO expects: prompt (str), completion (str), label (bool)
  """
  records = []
  with open(dataset_path, 'r') as f:
    for line in f:
      item = json.loads(line.strip())
      records.append(item)

  logging.info(f'Loaded {len(records)} records from {dataset_path}')

  n_positive = sum(1 for r in records if r['label'] == 1)
  n_negative = len(records) - n_positive
  logging.info(f'Positive (label=1): {n_positive}, Negative (label=0): {n_negative}')

  prompts = []
  completions = []
  labels = []

  for record in records:
    input_text = record['input_text']
    generated_text = record['generated_text']
    label = bool(record['label'])

    # Construct prompt using template
    if input_text:
      prompt = prompt_template.format(feature=input_text)
    else:
      # For unconditional generation, use template as-is
      prompt = prompt_template

    prompts.append(prompt)
    completions.append(generated_text)
    labels.append(label)

  dataset = Dataset.from_dict({
      'prompt': prompts,
      'completion': completions,
      'label': labels,
  })

  return dataset


def main():
  args = parse_args()

  access_token = args.access_token or hf_token

  logging.info(f'Model: {args.model_name_or_path}')
  logging.info(f'Dataset: {args.dataset_path}')
  logging.info(f'Output: {args.output_dir}')
  logging.info(f'Prompt style: {args.prompt_style}')

  os.makedirs(args.output_dir, exist_ok=True)

  # --- Load prompt template ---
  prompt_dict = get_prompt_dict(args.prompt_style)
  prompt_template = prompt_dict['prompt']
  logging.info(f'Prompt template: {prompt_template}')

  # --- Load dataset ---
  dataset = load_labeled_dataset(args.dataset_path, prompt_template)

  # --- Load model ---
  compute_dtype = (
      torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

  # Load training model (will have LoRA adapters added by KTOTrainer)
  logging.info('Loading training model...')
  model = AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      torch_dtype=compute_dtype,
      low_cpu_mem_usage=True,
      attn_implementation='eager',
      token=access_token,
  )

  # Load reference model explicitly.
  #
  # NOTE: ref_model=None with disable_adapter() was tried but results in KL=0
  # throughout training in multi-GPU DDP mode (confirmed from logs). We load
  # a separate frozen copy of the base model instead.
  # Memory: ~2GB per GPU for a 1B model in bfloat16 → total ~4GB/GPU with LoRA.
  logging.info('Loading reference model...')
  ref_model = AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      torch_dtype=compute_dtype,
      low_cpu_mem_usage=True,
      attn_implementation='eager',
      token=access_token,
  )
  for param in ref_model.parameters():
    param.requires_grad = False
  logging.info('Reference model loaded and frozen.')

  tokenizer = AutoTokenizer.from_pretrained(
      args.model_name_or_path,
      use_fast=False,
      token=access_token,
  )
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id

  # --- LoRA config ---
  peft_config = LoraConfig(
      r=args.lora_r,
      lora_alpha=args.lora_alpha,
      lora_dropout=args.lora_dropout,
      bias='none',
      task_type='CAUSAL_LM',
      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
  )

  # --- KTO config ---
  # Note: We'll manually save checkpoints after training to match train_clm.py format
  # So we set save_strategy='no' during training, then save manually
  kto_config = KTOConfig(
      output_dir=args.output_dir,
      learning_rate=args.learning_rate,
      num_train_epochs=args.num_train_epochs,
      max_steps=args.max_steps,
      per_device_train_batch_size=args.per_device_train_batch_size,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      warmup_steps=args.warmup_steps,
      lr_scheduler_type=args.lr_scheduler_type,
      beta=args.beta,
      max_prompt_length=args.max_prompt_length,
      max_completion_length=args.max_completion_length,
      logging_steps=args.logging_steps,
      save_strategy='no',  # We'll save manually after training
      seed=args.seed,
      bf16=args.bf16 and torch.cuda.is_bf16_supported(),
      fp16=not (args.bf16 and torch.cuda.is_bf16_supported()),
      gradient_checkpointing=args.gradient_checkpointing,
      gradient_checkpointing_kwargs={'use_reentrant': False},
      ddp_find_unused_parameters=False,
      report_to='wandb',
      run_name=args.output_dir.split('/')[-1],
  )

  # --- Trainer ---
  trainer = KTOTrainer(
      model=model,
      ref_model=ref_model,
      args=kto_config,
      train_dataset=dataset,
      peft_config=peft_config,
      processing_class=tokenizer,
  )

  logging.info('Starting KTO training...')
  trainer.train()
  logging.info('KTO training complete.')

  # --- Save model (following train_clm.py pattern) ---
  # Only rank 0 saves. All other ranks skip to avoid concurrent writes to the
  # same directory (race condition / file corruption).
  final_epoch = args.num_train_epochs - 1

  if trainer.is_world_process_zero():
    # Save LoRA adapter
    peft_save_dir = os.path.join(args.output_dir, f'peftmodel_epoch{final_epoch}')
    os.makedirs(peft_save_dir, exist_ok=True)
    trainer.model.save_pretrained(peft_save_dir)
    tokenizer.save_pretrained(peft_save_dir)
    logging.info(f'Saved LoRA adapter to {peft_save_dir}')

    # Merge LoRA into base model and save.
    # Reload from the saved adapter + original base model (same pattern as
    # merge_sft_peft.py) to avoid deepcopy of a GPU model.
    logging.info('Merging LoRA adapter into base model...')
    from peft import PeftModel
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        attn_implementation='eager',
        token=access_token,
    )
    peft_for_merge = PeftModel.from_pretrained(base_for_merge, peft_save_dir)
    merged_model = peft_for_merge.merge_and_unload()

    merged_dir = os.path.join(args.output_dir, f'model_epoch{final_epoch}')
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logging.info(f'Saved merged model to {merged_dir}')

    # Save results JSON (matching train_clm.py)
    results_file = os.path.join(args.output_dir, f'all_results_epoch{final_epoch}.json')
    with open(results_file, 'w') as f:
      json.dump({
          'final_epoch': final_epoch,
          'num_train_epochs': args.num_train_epochs,
          'learning_rate': args.learning_rate,
          'beta': args.beta,
      }, f)
    logging.info(f'Saved training results to {results_file}')


if __name__ == '__main__':
  main()
