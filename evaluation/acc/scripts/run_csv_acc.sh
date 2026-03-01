#!/bin/bash
# Evaluate next-token prediction accuracy using CSV files
# This script demonstrates how to use CSV format with custom column names

# --- Configuration ---
MODEL_NAME="google/gemma-3-1b"
SYNTHETIC_CSV="../../data/synthetic_data.csv"  # CSV with synthetic data
REAL_CSV="../../data/real_data.csv"  # CSV with real data
TEXT_COLUMN="generated_text"  # Column name in CSV containing text
OUTPUT_DIR="./results/csv_acc_eval"
MAX_SEQ_LENGTH=512
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
NUM_EPOCHS=3
SEED=42

# --- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --synthetic_csv)
            SYNTHETIC_CSV="$2"
            shift 2
            ;;
        --real_csv)
            REAL_CSV="$2"
            shift 2
            ;;
        --text_column)
            TEXT_COLUMN="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--synthetic_csv PATH] [--real_csv PATH] [--text_column COLUMN] [--output_dir DIR] [--epochs N]"
            exit 1
            ;;
    esac
done

# --- Validate input files ---
if [[ ! -f "${SYNTHETIC_CSV}" ]]; then
    echo "Error: Synthetic CSV file not found: ${SYNTHETIC_CSV}"
    exit 1
fi

if [[ ! -f "${REAL_CSV}" ]]; then
    echo "Error: Real CSV file not found: ${REAL_CSV}"
    exit 1
fi

# --- Create output directory ---
mkdir -p ${OUTPUT_DIR}

# --- Log configuration ---
echo "=========================================="
echo "CSV-based Accuracy Evaluation"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Synthetic CSV: ${SYNTHETIC_CSV}"
echo "Real CSV: ${REAL_CSV}"
echo "Text column: ${TEXT_COLUMN}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# --- Run evaluation ---
python ../compute_acc.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file ${SYNTHETIC_CSV} \
    --validation_file ${REAL_CSV} \
    --text_column ${TEXT_COLUMN} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps 50 \
    --save_total_limit 2 \
    --seed ${SEED} \
    --fp16 \
    --report_to "none"

# --- Display results ---
echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
if [[ -f "${OUTPUT_DIR}/eval_results.json" ]]; then
    echo "Results:"
    cat ${OUTPUT_DIR}/eval_results.json | python -m json.tool
fi
