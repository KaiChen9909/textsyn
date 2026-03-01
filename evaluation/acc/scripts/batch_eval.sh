#!/bin/bash
# Batch evaluation script for comparing multiple synthetic datasets or models
# This script evaluates multiple synthetic datasets against the same real data

set -e

echo "=========================================="
echo "Batch Accuracy Evaluation"
echo "=========================================="

# --- Configuration ---
REAL_DATA="../../data/real_biorxiv.txt"  # Real validation data (same for all)
MODEL_NAME="google/gemma-3-1b"
BASE_OUTPUT_DIR="./results/batch_eval"
NUM_EPOCHS=3
BATCH_SIZE=8
MAX_TRAIN_SAMPLES=5000  # Limit for faster evaluation
MAX_EVAL_SAMPLES=1000

# --- Synthetic datasets to evaluate ---
# Add your synthetic datasets here
SYNTHETIC_DATASETS=(
    "../../data/synthetic_baseline.txt:baseline"
    "../../data/synthetic_dp_eps4.txt:dp_eps4"
    "../../data/synthetic_dp_eps8.txt:dp_eps8"
    "../../data/synthetic_nondp.txt:nondp"
)

# --- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --real_data)
            REAL_DATA="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --max_train)
            MAX_TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --max_eval)
            MAX_EVAL_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--real_data PATH] [--epochs N] [--max_train N] [--max_eval N]"
            exit 1
            ;;
    esac
done

# --- Validate real data ---
if [[ ! -f "${REAL_DATA}" ]]; then
    echo "Error: Real data file not found: ${REAL_DATA}"
    echo "Please update REAL_DATA path or use --real_data option"
    exit 1
fi

# --- Create base output directory ---
mkdir -p ${BASE_OUTPUT_DIR}

# --- Summary file ---
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary.json"
echo "[" > ${SUMMARY_FILE}

# --- Run evaluation for each synthetic dataset ---
FIRST=true
for entry in "${SYNTHETIC_DATASETS[@]}"; do
    # Parse entry: path:name
    IFS=':' read -r SYNTHETIC_DATA DATASET_NAME <<< "$entry"

    if [[ ! -f "${SYNTHETIC_DATA}" ]]; then
        echo "Warning: Synthetic data not found: ${SYNTHETIC_DATA}, skipping..."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: ${DATASET_NAME}"
    echo "=========================================="
    echo "Synthetic data: ${SYNTHETIC_DATA}"
    echo "Real data: ${REAL_DATA}"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATASET_NAME}"
    mkdir -p ${OUTPUT_DIR}

    # Run evaluation
    python ../compute_acc.py \
        --model_name_or_path ${MODEL_NAME} \
        --train_file ${SYNTHETIC_DATA} \
        --validation_file ${REAL_DATA} \
        --text_column "text" \
        --max_seq_length 512 \
        --max_train_samples ${MAX_TRAIN_SAMPLES} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
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
        --save_total_limit 1 \
        --fp16 \
        --seed 42 \
        --report_to "none"

    # Extract metrics and add to summary
    if [[ -f "${OUTPUT_DIR}/eval_results.json" ]]; then
        if [ "$FIRST" = false ]; then
            echo "," >> ${SUMMARY_FILE}
        fi
        FIRST=false

        echo "  {" >> ${SUMMARY_FILE}
        echo "    \"name\": \"${DATASET_NAME}\"," >> ${SUMMARY_FILE}
        echo "    \"synthetic_data\": \"${SYNTHETIC_DATA}\"," >> ${SUMMARY_FILE}

        # Extract metrics
        ACCURACY=$(cat ${OUTPUT_DIR}/eval_results.json | python -c "import sys, json; print(json.load(sys.stdin)['eval_accuracy'])")
        LOSS=$(cat ${OUTPUT_DIR}/eval_results.json | python -c "import sys, json; print(json.load(sys.stdin)['eval_loss'])")
        PERPLEXITY=$(cat ${OUTPUT_DIR}/eval_results.json | python -c "import sys, json; print(json.load(sys.stdin).get('perplexity', 'N/A'))")

        echo "    \"accuracy\": ${ACCURACY}," >> ${SUMMARY_FILE}
        echo "    \"loss\": ${LOSS}," >> ${SUMMARY_FILE}
        echo "    \"perplexity\": ${PERPLEXITY}" >> ${SUMMARY_FILE}
        echo "  }" >> ${SUMMARY_FILE}

        echo "Results for ${DATASET_NAME}:"
        echo "  Accuracy: ${ACCURACY}"
        echo "  Loss: ${LOSS}"
        echo "  Perplexity: ${PERPLEXITY}"
    fi
done

echo "]" >> ${SUMMARY_FILE}

# --- Display final summary ---
echo ""
echo "=========================================="
echo "Batch Evaluation Complete!"
echo "=========================================="
echo ""
echo "Summary of all evaluations:"
echo ""

python -c "
import json
import sys

with open('${SUMMARY_FILE}', 'r') as f:
    results = json.load(f)

# Sort by accuracy (descending)
results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

print(f\"{'Rank':<6} {'Dataset':<25} {'Accuracy':<12} {'Loss':<12} {'Perplexity':<12}\")
print('-' * 75)

for i, result in enumerate(results_sorted, 1):
    print(f\"{i:<6} {result['name']:<25} {result['accuracy']:<12.4f} {result['loss']:<12.4f} {str(result['perplexity']):<12}\")

print()
print(f\"Best performing: {results_sorted[0]['name']} (Accuracy: {results_sorted[0]['accuracy']:.4f})\")
" 2>/dev/null || cat ${SUMMARY_FILE}

echo ""
echo "Detailed results saved to: ${BASE_OUTPUT_DIR}"
echo "Summary saved to: ${SUMMARY_FILE}"
echo ""
