#!/bin/bash

# ==========================================
# Sample Topics from Distribution
# ==========================================
# This script samples topics from a saved topic distribution
# and generates keyword conditions for text generation.

# ==========================================
# Configuration Parameters
# ==========================================
DATASET_NAME=${1:-biorxiv}
RHO=${2:-0.18}

DISTRIBUTION_PATH="./distribution/${DATASET_NAME}_topic_distribution_train.json"
OUTPUT_PATH="./results/${DATASET_NAME}_sampled_topics.csv"
N_SAMPLES=5000
FORMAT="csv"  # Options: csv, jsonl
SEED=42

# ==========================================
# Environment Setup
# ==========================================
mkdir -p results

# load anaconda and activate env (if needed)
# module load anaconda
# source activate syn

cd $SLURM_SUBMIT_DIR
cd ..  # Move to TOPIC directory

# ==========================================
# Run Sampling
# ==========================================
echo "=== Sampling Topics ==="
echo "Distribution: $DISTRIBUTION_PATH"
echo "Output: $OUTPUT_PATH"
echo "Number of samples: $N_SAMPLES"
echo "Sampling mode: $SAMPLING_MODE"
echo "Format: $FORMAT"
echo "Random seed: $SEED"
echo ""

python sample_topics.py \
    --distribution_path ${DISTRIBUTION_PATH} \
    --output_file ${OUTPUT_PATH} \
    --n_samples ${N_SAMPLES} \
    --rho ${RHO} \
    --format ${FORMAT} \
    --seed ${SEED}

echo ""
echo "=== Sampling Complete ==="
echo "Output saved to: ${OUTPUT_PATH}"
echo ""
echo "You can now use this file for generation with:"
echo "  cd ../DPSFT"
echo "  python generation_biorxiv_condgen.py \\"
echo "    --prompt_file ${OUTPUT_PATH} \\"
echo "    --model_name_or_path <your_trained_model> \\"
echo "    --output_dir ${DATASET_NAME}_sampled \\"
echo "    --schema_column schema"
