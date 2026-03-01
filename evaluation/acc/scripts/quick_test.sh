#!/bin/bash
# Quick test script to verify the setup and run a small test evaluation
# This script uses a small subset of data for rapid testing

set -e

echo "=========================================="
echo "Quick Test for Accuracy Evaluation"
echo "=========================================="
echo "This script will run a quick test to verify your setup."
echo ""

# --- Configuration ---
MODEL_NAME="google/gemma-3-1b"  # Small model for quick testing
MAX_TRAIN_SAMPLES=100  # Very small for quick test
MAX_EVAL_SAMPLES=50
NUM_EPOCHS=1
BATCH_SIZE=4
OUTPUT_DIR="./results/quick_test"

# --- Parse command line arguments ---
TRAIN_FILE=""
EVAL_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --eval)
            EVAL_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 --train TRAIN_FILE --eval EVAL_FILE [--model MODEL]"
            echo ""
            echo "Options:"
            echo "  --train FILE   Path to training data (synthetic)"
            echo "  --eval FILE    Path to evaluation data (real)"
            echo "  --model MODEL  Model name or path (default: google/gemma-3-1b)"
            exit 1
            ;;
    esac
done

# --- Validate arguments ---
if [[ -z "${TRAIN_FILE}" ]] || [[ -z "${EVAL_FILE}" ]]; then
    echo "Error: Both --train and --eval arguments are required"
    echo ""
    echo "Example:"
    echo "  $0 --train synthetic.txt --eval real.txt"
    exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Error: Training file not found: ${TRAIN_FILE}"
    exit 1
fi

if [[ ! -f "${EVAL_FILE}" ]]; then
    echo "Error: Evaluation file not found: ${EVAL_FILE}"
    exit 1
fi

# --- Check dependencies ---
echo "Checking dependencies..."
python -c "import transformers; import datasets; import evaluate; import torch" 2>/dev/null || {
    echo "Error: Required packages not found. Please install:"
    echo "  pip install -r ../requirements.txt"
    exit 1
}
echo "✓ Dependencies OK"
echo ""

# --- Display configuration ---
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Training data: ${TRAIN_FILE}"
echo "  Evaluation data: ${EVAL_FILE}"
echo "  Max train samples: ${MAX_TRAIN_SAMPLES}"
echo "  Max eval samples: ${MAX_EVAL_SAMPLES}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# --- Create output directory ---
mkdir -p ${OUTPUT_DIR}

# --- Run quick test ---
echo "Running quick test evaluation..."
echo "This may take a few minutes depending on your hardware..."
echo ""

python ../compute_acc.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file ${TRAIN_FILE} \
    --validation_file ${EVAL_FILE} \
    --text_column "text" \
    --max_seq_length 256 \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_strategy "no" \
    --evaluation_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --fp16 \
    --seed 42 \
    --report_to "none"

# --- Display results ---
echo ""
echo "=========================================="
echo "Quick Test Complete!"
echo "=========================================="

if [[ -f "${OUTPUT_DIR}/eval_results.json" ]]; then
    echo ""
    echo "✓ Test successful! Your setup is working correctly."
    echo ""
    echo "Results:"
    cat ${OUTPUT_DIR}/eval_results.json | python -m json.tool

    ACCURACY=$(cat ${OUTPUT_DIR}/eval_results.json | python -c "import sys, json; print(json.load(sys.stdin)['eval_accuracy'])")
    echo ""
    echo "Next-token prediction accuracy: ${ACCURACY}"
    echo ""
    echo "You can now run full evaluations using the other scripts in this directory."
else
    echo "✗ Test failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "To run a full evaluation, use:"
echo "  bash run_biorxiv_acc.sh --synthetic_data <path> --real_data <path>"
echo ""
