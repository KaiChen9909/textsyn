# Next-Token Prediction Accuracy Evaluation

This module evaluates the utility of synthetic text data by training a language model on synthetic data and measuring next-token prediction accuracy on real data.

## Overview

The evaluation follows this workflow:
1. **Train** a language model on synthetic data
2. **Evaluate** the model's next-token prediction accuracy on real data
3. **Compute** metrics including accuracy and perplexity

This approach measures how well synthetic data captures the patterns in real data for language modeling tasks.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a quick test
cd scripts
bash quick_test.sh --train /path/to/synthetic.txt --eval /path/to/real.txt

# 3. Run full evaluation
bash run_biorxiv_acc.sh \
    --synthetic_data /path/to/synthetic.txt \
    --real_data /path/to/real.txt \
    --epochs 3
```

## Installation

Install the required dependencies:

```bash
pip install transformers datasets evaluate torch accelerate
# Or use the requirements file
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python compute_acc.py \
    --model_name_or_path google/gemma-3-1b \
    --train_file /path/to/synthetic_data.txt \
    --validation_file /path/to/real_data.txt \
    --output_dir ./results/acc_eval \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 5e-5
```

### Using Shell Scripts

For convenience, use the provided shell scripts:

```bash
# Example 1: Using the general script
bash eval_acc.sh

# Example 2: Using the BioRxiv-specific script
cd scripts
bash run_biorxiv_acc.sh \
    --model google/gemma-3-1b \
    --synthetic_data /path/to/synthetic.txt \
    --real_data /path/to/real.txt \
    --output_dir ./results/my_eval \
    --epochs 3 \
    --batch_size 8
```

## Arguments

### Model Arguments

- `--model_name_or_path`: Path to pretrained model or HuggingFace model ID (required)
- `--config_name`: Pretrained config name or path (optional)
- `--tokenizer_name`: Pretrained tokenizer name or path (optional)
- `--torch_dtype`: Model dtype (auto, bfloat16, float16, float32)
- `--low_cpu_mem_usage`: Enable memory-efficient model loading

### Data Arguments

- `--train_file`: Path to synthetic training data (required)
- `--validation_file`: Path to real validation data (required)
- `--text_column`: Column name containing text data (default: "text")
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--max_train_samples`: Limit training samples for quick evaluation
- `--max_eval_samples`: Limit evaluation samples for quick evaluation
- `--preprocessing_num_workers`: Number of preprocessing workers

### Training Arguments

- `--output_dir`: Output directory for model and results (required)
- `--do_train`: Whether to run training
- `--do_eval`: Whether to run evaluation
- `--per_device_train_batch_size`: Training batch size per device (default: 8)
- `--per_device_eval_batch_size`: Evaluation batch size per device (default: 8)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--evaluation_strategy`: Evaluation strategy (steps, epoch)
- `--save_strategy`: Save strategy (steps, epoch)
- `--fp16`: Enable mixed precision training
- `--seed`: Random seed for reproducibility (default: 42)

## Input Data Formats

The script supports multiple input formats:

### Text Files (.txt)

One document per line:
```
This is document 1.
This is document 2.
This is document 3.
```

### JSON Files (.json)

```json
{"text": "This is document 1."}
{"text": "This is document 2."}
{"text": "This is document 3."}
```

### CSV Files (.csv)

```csv
text
"This is document 1."
"This is document 2."
"This is document 3."
```

## Output

The evaluation produces the following outputs in the specified `output_dir`:

1. **eval_results.json**: Evaluation metrics including:
   - `eval_accuracy`: Next-token prediction accuracy
   - `eval_loss`: Cross-entropy loss
   - `perplexity`: Perplexity score (exp of loss)
   - `eval_samples`: Number of evaluation samples

2. **all_results.json**: Complete training and evaluation metrics

3. **checkpoint-*/**: Model checkpoints saved during training

4. **trainer_state.json**: Training state for resuming

## Example Output

```json
{
  "eval_loss": 2.3456,
  "eval_accuracy": 0.4523,
  "perplexity": 10.44,
  "eval_runtime": 120.5,
  "eval_samples": 1000,
  "epoch": 3.0
}
```

## Advanced Usage

### Training on CSV with Custom Column

```bash
python compute_acc.py \
    --model_name_or_path gpt2 \
    --train_file synthetic_data.csv \
    --validation_file real_data.csv \
    --text_column "abstract" \
    --output_dir ./results \
    --do_train --do_eval
```

### Quick Evaluation with Sample Limits

```bash
python compute_acc.py \
    --model_name_or_path google/gemma-3-1b \
    --train_file synthetic.txt \
    --validation_file real.txt \
    --max_train_samples 1000 \
    --max_eval_samples 500 \
    --output_dir ./results/quick_test \
    --do_train --do_eval \
    --num_train_epochs 1
```

### Resume from Checkpoint

```bash
python compute_acc.py \
    --model_name_or_path google/gemma-3-1b \
    --train_file synthetic.txt \
    --validation_file real.txt \
    --output_dir ./results/checkpoint \
    --resume_from_checkpoint ./results/checkpoint/checkpoint-500 \
    --do_train --do_eval
```

## Notes

- The script automatically handles padding tokens by masking them during metric computation
- Next-token prediction accuracy is computed by shifting predictions and labels
- Perplexity is calculated as `exp(loss)` to measure model uncertainty
- Use `--fp16` for faster training on compatible GPUs
- Adjust `--gradient_accumulation_steps` if you run out of memory

## Citation

This implementation is based on the approach from:
```
AI-secure/aug-pe: https://github.com/AI-secure/aug-pe/blob/main/utility_eval/run_clm.py
```

## References

- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- HuggingFace Datasets: https://huggingface.co/docs/datasets/
- Evaluate Library: https://huggingface.co/docs/evaluate/
