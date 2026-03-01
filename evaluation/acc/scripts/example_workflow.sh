#!/bin/bash
# Complete workflow example for next-token prediction accuracy evaluation
# This script demonstrates the full pipeline from data preparation to evaluation

set -e  # Exit on error

echo "=========================================="
echo "Next-Token Prediction Accuracy Workflow"
echo "=========================================="

# --- Configuration ---
# Adjust these paths to your actual data
SYNTHETIC_CSV="../../data/synthetic_biorxiv.csv"
REAL_CSV="../../data/real_biorxiv.csv"
TEXT_COLUMN="abstract"  # or "generated_text", depending on your data

# Temporary converted files
SYNTHETIC_TXT="./data/synthetic_converted.txt"
REAL_TXT="./data/real_converted.txt"

# Model and training settings
MODEL_NAME="google/gemma-3-1b"
OUTPUT_DIR="./results/example_workflow"
NUM_EPOCHS=3
BATCH_SIZE=8

# --- Step 0: Create data directory ---
mkdir -p ./data

# --- Step 1: Inspect data ---
echo ""
echo "Step 1: Inspecting data files..."
echo "---"

if [[ -f "${SYNTHETIC_CSV}" ]]; then
    echo "Inspecting synthetic data:"
    python ../prepare_data.py inspect \
        --input ${SYNTHETIC_CSV} \
        --format csv \
        --text_column ${TEXT_COLUMN} \
        --num_samples 3
else
    echo "Warning: Synthetic CSV not found at ${SYNTHETIC_CSV}"
    echo "Please update the path in this script."
fi

echo ""

if [[ -f "${REAL_CSV}" ]]; then
    echo "Inspecting real data:"
    python ../prepare_data.py inspect \
        --input ${REAL_CSV} \
        --format csv \
        --text_column ${TEXT_COLUMN} \
        --num_samples 3
else
    echo "Warning: Real CSV not found at ${REAL_CSV}"
    echo "Please update the path in this script."
fi

# --- Step 2: Convert data to TXT format ---
echo ""
echo "Step 2: Converting data to TXT format..."
echo "---"

if [[ -f "${SYNTHETIC_CSV}" ]]; then
    python ../prepare_data.py convert \
        --input ${SYNTHETIC_CSV} \
        --output ${SYNTHETIC_TXT} \
        --input_format csv \
        --text_column ${TEXT_COLUMN}

    echo "Synthetic data converted: ${SYNTHETIC_TXT}"
    echo "Number of lines: $(wc -l < ${SYNTHETIC_TXT})"
fi

if [[ -f "${REAL_CSV}" ]]; then
    python ../prepare_data.py convert \
        --input ${REAL_CSV} \
        --output ${REAL_TXT} \
        --input_format csv \
        --text_column ${TEXT_COLUMN}

    echo "Real data converted: ${REAL_TXT}"
    echo "Number of lines: $(wc -l < ${REAL_TXT})"
fi

# --- Step 3: Run accuracy evaluation ---
echo ""
echo "Step 3: Running accuracy evaluation..."
echo "---"
echo "Training on synthetic data: ${SYNTHETIC_TXT}"
echo "Evaluating on real data: ${REAL_TXT}"
echo ""

if [[ -f "${SYNTHETIC_TXT}" ]] && [[ -f "${REAL_TXT}" ]]; then
    python ../compute_acc.py \
        --model_name_or_path ${MODEL_NAME} \
        --train_file ${SYNTHETIC_TXT} \
        --validation_file ${REAL_TXT} \
        --text_column "text" \
        --max_seq_length 512 \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --num_train_epochs ${NUM_EPOCHS} \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --logging_strategy "steps" \
        --logging_steps 50 \
        --save_total_limit 2 \
        --fp16 \
        --seed 42 \
        --report_to "none"
else
    echo "Error: Converted data files not found. Skipping evaluation."
    exit 1
fi

# --- Step 4: Display results ---
echo ""
echo "=========================================="
echo "Step 4: Results Summary"
echo "=========================================="

if [[ -f "${OUTPUT_DIR}/eval_results.json" ]]; then
    echo ""
    echo "Evaluation Metrics:"
    cat ${OUTPUT_DIR}/eval_results.json | python -m json.tool

    echo ""
    echo "Key Metrics:"
    ACCURACY=$(cat ${OUTPUT_DIR}/eval_results.json | python -c "import sys, json; print(json.load(sys.stdin)['eval_accuracy'])")
    PERPLEXITY=$(cat ${OUTPUT_DIR}/eval_results.json | python -c "import sys, json; print(json.load(sys.stdin).get('perplexity', 'N/A'))")

    echo "  Next-Token Prediction Accuracy: ${ACCURACY}"
    echo "  Perplexity: ${PERPLEXITY}"
fi

echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Files generated:"
echo "  - ${OUTPUT_DIR}/eval_results.json"
echo "  - ${OUTPUT_DIR}/all_results.json"
echo "  - ${OUTPUT_DIR}/checkpoint-*/"
echo ""
