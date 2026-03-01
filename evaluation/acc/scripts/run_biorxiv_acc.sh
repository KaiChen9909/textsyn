#!/bin/bash
# Evaluate next-token prediction accuracy for BioRxiv dataset
# Train on synthetic data and evaluate on real data

# --- Configuration ---
MODEL_NAME="google/gemma-3-1b"
SYNTHETIC_DATA="../../data/synthetic_biorxiv.txt"  # Adjust path to your synthetic data
REAL_DATA="../../data/real_biorxiv.txt"  # Adjust path to your real data
OUTPUT_DIR="./results/biorxiv_acc_${MODEL_NAME//\//_}"
TEXT_COLUMN="text"
MAX_SEQ_LENGTH=512
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
NUM_EPOCHS=3
SEED=42
MAX_TRAIN_SAMPLES=10000  # Optional: limit training samples for quick evaluation
MAX_EVAL_SAMPLES=1000    # Optional: limit eval samples for quick evaluation

# --- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --synthetic_data)
            SYNTHETIC_DATA="$2"
            shift 2
            ;;
        --real_data)
            REAL_DATA="$2"
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
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_train_samples)
            MAX_TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --max_eval_samples)
            MAX_EVAL_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- Validate required arguments ---
if [[ ! -f "${SYNTHETIC_DATA}" ]]; then
    echo "Error: Synthetic data file not found: ${SYNTHETIC_DATA}"
    exit 1
fi

if [[ ! -f "${REAL_DATA}" ]]; then
    echo "Error: Real data file not found: ${REAL_DATA}"
    exit 1
fi

# --- Create output directory ---
mkdir -p ${OUTPUT_DIR}

# --- Log configuration ---
echo "=========================================="
echo "Next-Token Prediction Accuracy Evaluation"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Synthetic data: ${SYNTHETIC_DATA}"
echo "Real data: ${REAL_DATA}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Max sequence length: ${MAX_SEQ_LENGTH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Number of epochs: ${NUM_EPOCHS}"
echo "=========================================="

# --- Run evaluation ---
python ../compute_acc.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file ${SYNTHETIC_DATA} \
    --validation_file ${REAL_DATA} \
    --text_column ${TEXT_COLUMN} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
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
echo "Results saved to: ${OUTPUT_DIR}/eval_results.json"
echo ""

if [[ -f "${OUTPUT_DIR}/eval_results.json" ]]; then
    echo "Evaluation metrics:"
    cat ${OUTPUT_DIR}/eval_results.json
fi
