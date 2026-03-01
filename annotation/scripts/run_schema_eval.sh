#!/bin/bash

# Example script to run schema_eval.py
# Evaluates schema-based text generation quality using BERTScore

# Configuration
DATA_PATH="../data/biorxiv/clean_biorxiv_schema_noexample_train.csv"
MODEL_NAME="google/gemma-3-1b-pt"
TEXT_COLUMN="abstract"
SAMPLE_SIZE=100  # Set to evaluate on a subset, remove for full dataset
DEVICE="cuda:0"
OUTPUT_PATH="results/schema_eval_results.json"
SEED=42

# Generation parameters
BATCH_SIZE=8
MAX_PROMPT_LENGTH=512
MAX_GEN_LENGTH=512
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=0

# BERTScore parameters
BERTSCORE_BATCH_SIZE=32

python schema_eval.py \
    --data_path ${DATA_PATH} \
    --model_name_or_path ${MODEL_NAME} \
    --text_column ${TEXT_COLUMN} \
    --sample_size ${SAMPLE_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --max_gen_length ${MAX_GEN_LENGTH} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --device ${DEVICE} \
    --output_path ${OUTPUT_PATH} \
    --seed ${SEED} \
    --bertscore_batch_size ${BERTSCORE_BATCH_SIZE} \
    --save_generations
