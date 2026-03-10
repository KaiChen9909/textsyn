#!/bin/bash

# ==========================================
# SLURM Resource Configuration (optional - uncomment if submitting as SLURM job)
# ==========================================
#SBATCH --job-name=extract_topics
#SBATCH --account=CIS260108-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_extract_topics_stdout.txt

# ==========================================
# Environment Setup
# ==========================================
module load anaconda
source activate syn

cd $SLURM_SUBMIT_DIR

# ==========================================
# Parse Arguments
# ==========================================
DATASET_NAME=${1:-biorxiv}

# ==========================================
# Dataset-specific Configuration
# ==========================================
case $DATASET_NAME in
    biorxiv)
        DATA_DIR="../data/biorxiv"
        TEXT_COLUMN="abstract"
        TRAIN_FILE="train.csv"
        VALID_FILE="valid.csv"
        ;;
    openreview)
        DATA_DIR="../data/openreview"
        TEXT_COLUMN="text"
        TRAIN_FILE="train.csv"
        VALID_FILE="valid.csv"
        ;;
    *)
        echo "Error: Unknown dataset '${DATASET_NAME}'"
        echo ""
        echo "Supported datasets:"
        echo "  - biorxiv"
        echo "  - openreview"
        echo ""
        echo "Usage: bash extract_topics.sh <dataset_name>"
        exit 1
        ;;
esac

# ==========================================
# Common Configuration
# ==========================================
EMBEDDING_MODEL="all-MiniLM-L6-v2"  # Sentence transformer model
MODEL_PATH="./models/ctcl_topic"

# ==========================================
# Check Data Availability
# ==========================================
if [ ! -f "${DATA_DIR}/${TRAIN_FILE}" ]; then
    echo "ERROR: Training data not found at: ${DATA_DIR}/${TRAIN_FILE}"
    exit 1
fi

if [ ! -f "${DATA_DIR}/${VALID_FILE}" ]; then
    echo "ERROR: Validation data not found at: ${DATA_DIR}/${VALID_FILE}"
    exit 1
fi

# ==========================================
# Run Topic Extraction
# ==========================================
echo ""
echo "=== Configuration ==="
echo "Dataset name: $DATASET_NAME"
echo "Data directory: $DATA_DIR"
echo "Text column: $TEXT_COLUMN"
echo "Training file: ${DATA_DIR}/${TRAIN_FILE}"
echo "Validation file: ${DATA_DIR}/${VALID_FILE}"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Pretrained model: $MODEL_PATH"
echo ""

# ==========================================
# Step 1: Extract topics for training set
# ==========================================
echo "=== Step 1: Extracting topics for training set ==="

python extract_topics.py \
    --input_file ${DATA_DIR}/${TRAIN_FILE} \
    --output_file ${DATA_DIR}/clean_${DATASET_NAME}_topic_train.csv \
    --dataset_name ${DATASET_NAME} \
    --text_column ${TEXT_COLUMN} \
    --model_path ${MODEL_PATH} \
    --embedding_model ${EMBEDDING_MODEL} \
    --distribution_path ./distribution/${DATASET_NAME}_topic_distribution_train.json

echo "Training set topic extraction complete!"
echo ""

# ==========================================
# Step 2: Extract topics for validation set
# ==========================================
echo "=== Step 2: Extracting topics for validation set ==="

python extract_topics.py \
    --input_file ${DATA_DIR}/${VALID_FILE} \
    --output_file ${DATA_DIR}/clean_${DATASET_NAME}_topic_valid.csv \
    --dataset_name ${DATASET_NAME} \
    --text_column ${TEXT_COLUMN} \
    --model_path ${MODEL_PATH} \
    --embedding_model ${EMBEDDING_MODEL}

echo "Validation set topic extraction complete!"
echo ""

# ==========================================
# Completion
# ==========================================
echo "=== All steps completed successfully! ==="
echo ""
echo "Output files:"
echo "  Training data: ${DATA_DIR}/clean_${DATASET_NAME}_topic_train.csv"
echo "  Validation data: ${DATA_DIR}/clean_${DATASET_NAME}_topic_valid.csv"
echo "  Topic distribution: ./distribution/${DATASET_NAME}_topic_distribution_train.json"
echo ""
echo "Job finished at: $(date)"
