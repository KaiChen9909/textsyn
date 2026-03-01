#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_gen
#SBATCH --account=NAIRR250463-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_infer_schema_stdout.txt

# ==========================================
# 环境准备
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load anaconda
source activate syn

cd $SLURM_SUBMIT_DIR

# ==========================================
# 运行你的 Bash 脚本任务
# ==========================================
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

DATASET_NAME=${1:-biorxiv}
SCHEMA_TYPE=${2:-noexample}
NUM_FEATURES=${3:-8}
TEXT_COLUMN=${4:-abstract}

echo "inferring schema"

python infer_biorxiv_schema.py \
    --input_file ../data/${DATASET_NAME}/train.csv \
    --output_file ../data/${DATASET_NAME}/${DATASET_NAME}_schema_${SCHEMA_TYPE}_train.csv \
    --schema_name ${DATASET_NAME}_schema_${SCHEMA_TYPE} \
    --text_column ${TEXT_COLUMN}

python infer_biorxiv_schema.py \
    --input_file ../data/${DATASET_NAME}/validation.csv \
    --output_file ../data/${DATASET_NAME}/${DATASET_NAME}_schema_${SCHEMA_TYPE}_valid.csv \
    --schema_name ${DATASET_NAME}_schema_${SCHEMA_TYPE} \
    --text_column ${TEXT_COLUMN}

echo "inference finished"