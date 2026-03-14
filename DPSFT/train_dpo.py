"""DPO training script - Alternative to KTO without ref_model memory issues.

DPO (Direct Preference Optimization) directly optimizes on preference pairs
without needing an explicit reference model during training.
"""

import argparse
import copy
import json
import logging
import os

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from utils.data_utils import get_prompt_dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

load_dotenv('../.env')
hf_token = os.getenv('HF_LOGIN_STR')


def parse_args():
  parser = argparse.ArgumentParser(description='DPO training on labeled generation data.')
  
  # Model
  parser.add_argument('--model_name_or_path', type=str, required=True)
  parser.add_argument('--access_token', type=str, default=None)
  
  # Data
  parser.add_argument('--dataset_path', type=str, required=True)
  parser.add_argument('--prompt_style', type=str, default='biorxiv_condgen_filter_generation')
  
  # Output
  parser.add_argument('--output_dir', type=str, required=True)
  
  # Training hyperparameters
  parser.add_argument('--learning_rate', type=float, default=5e-6)
  parser.add_argument('--num_train_epochs', type=int, default=1)
  parser.add_argument('--per_device_train_batch_size', type=int, default=4)
  parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
  parser.add_argument('--warmup_steps', type=int, default=10)
  parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
  
  # DPO specific
  parser.add_argument('--beta', type=float, default=0.1, help='DPO beta parameter')
  parser.add_argument('--max_prompt_length', type=int, default=300)
  parser.add_argument('--max_length', type=int, default=812)  # prompt + completion
  
  # LoRA
  parser.add_argument('--lora_r', type=int, default=8)
  parser.add_argument('--lora_alpha', type=float, default=16)
  parser.add_argument('--lora_dropout', type=float, default=0.05)
  
  # Misc
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--logging_steps', type=int, default=5)
  parser.add_argument('--gradient_checkpointing', action='store_true')
  parser.add_argument('--bf16', action='store_true', default=True)
  
  return parser.parse_args()


def convert_kto_to_dpo(dataset_path, prompt_template):
  """Convert KTO format to DPO format by pairing chosen/rejected samples."""
  records = []
  with open(dataset_path, 'r') as f:
    for line in f:
      records.append(json.loads(line.strip()))
  
  logging.info(f'Loaded {len(records)} KTO records')
  
  # Group by input_text
  from collections import defaultdict
  groups = defaultdict(lambda: {'chosen': [], 'rejected': []})
  
  for record in records:
    key = record['input_text']
    if record['label'] == 1:
      groups[key]['chosen'].append(record['generated_text'])
    else:
      groups[key]['rejected'].append(record['generated_text'])
  
  # Create DPO pairs
  dpo_data = []
  for input_text, samples in groups.items():
    chosen_list = samples['chosen']
    rejected_list = samples['rejected']
    
    # Create all possible pairs
    for chosen in chosen_list:
      for rejected in rejected_list:
        prompt = prompt_template.format(feature=input_text) if input_text else prompt_template
        dpo_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
        })
  
  logging.info(f'Created {len(dpo_data)} DPO pairs from {len(groups)} prompts')
  
  return Dataset.from_list(dpo_data)


def main():
  args = parse_args()
  
  access_token = args.access_token or hf_token
  
  logging.info(f'Model: {args.model_name_or_path}')
  logging.info(f'Dataset: {args.dataset_path}')
  logging.info(f'Output: {args.output_dir}')
  
  os.makedirs(args.output_dir, exist_ok=True)
  
  # Load prompt template
  prompt_dict = get_prompt_dict(args.prompt_style)
  prompt_template = prompt_dict['prompt']
  
  # Convert KTO to DPO format
  dataset = convert_kto_to_dpo(args.dataset_path, prompt_template)
  
  # Load model
  compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  
  model = AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      torch_dtype=compute_dtype,
      low_cpu_mem_usage=True,
      attn_implementation='eager',
      token=access_token,
  )
  
  tokenizer = AutoTokenizer.from_pretrained(
      args.model_name_or_path,
      use_fast=False,
      token=access_token,
  )
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id
  
  # LoRA config
  peft_config = LoraConfig(
      r=args.lora_r,
      lora_alpha=args.lora_alpha,
      lora_dropout=args.lora_dropout,
      bias='none',
      task_type='CAUSAL_LM',
      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
  )
  
  # DPO config
  dpo_config = DPOConfig(
      output_dir=args.output_dir,
      learning_rate=args.learning_rate,
      num_train_epochs=args.num_train_epochs,
      per_device_train_batch_size=args.per_device_train_batch_size,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      warmup_steps=args.warmup_steps,
      lr_scheduler_type=args.lr_scheduler_type,
      beta=args.beta,
      max_prompt_length=args.max_prompt_length,
      max_length=args.max_length,
      logging_steps=args.logging_steps,
      save_strategy='no',
      seed=args.seed,
      bf16=args.bf16 and torch.cuda.is_bf16_supported(),
      fp16=not (args.bf16 and torch.cuda.is_bf16_supported()),
      gradient_checkpointing=args.gradient_checkpointing,
      gradient_checkpointing_kwargs={'use_reentrant': False},
      report_to='wandb',
      run_name=args.output_dir.split('/')[-1],
  )
  
  # DPOTrainer - no ref_model needed!
  trainer = DPOTrainer(
      model=model,
      ref_model=None,  # DPO works without explicit ref_model!
      args=dpo_config,
      train_dataset=dataset,
      peft_config=peft_config,
      processing_class=tokenizer,
  )
  
  logging.info('Starting DPO training...')
  trainer.train()
  logging.info('DPO training complete.')
  
  # Save model
  final_epoch = args.num_train_epochs - 1
  
  peft_save_dir = os.path.join(args.output_dir, f'peftmodel_epoch{final_epoch}')
  os.makedirs(peft_save_dir, exist_ok=True)
  trainer.model.save_pretrained(peft_save_dir)
  tokenizer.save_pretrained(peft_save_dir)
  
  logging.info('Merging LoRA adapter...')
  merged_model = trainer.model.merge_and_unload()
  merged_dir = os.path.join(args.output_dir, f'model_epoch{final_epoch}')
  os.makedirs(merged_dir, exist_ok=True)
  merged_model.save_pretrained(merged_dir)
  tokenizer.save_pretrained(merged_dir)
  
  logging.info(f'Saved to {merged_dir}')


if __name__ == '__main__':
  main()
