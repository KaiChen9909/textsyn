#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH -J biorxiv_gen
#SBATCH -A dplab
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%j_infer_schema_stdout.txt

# ==========================================
# 环境准备
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load uv
source /scratch/pkq2ps/envs/syn/bin/activate

cd $SLURM_SUBMIT_DIR

# ==========================================
# 运行你的 Bash 脚本任务
# ==========================================
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"

DATASET_NAME=${1:-biorxiv}
SCHEMA_TYPE=${2:-noexample}
NUM_FEATURES=${3:-8}
TEXT_COLUMN=${4:-abstract}
SPLIT=${5:-train}

echo "inferring schema"

python infer_biorxiv_schema.py \
    --input_file ../data/${DATASET_NAME}/${SPLIT}.csv \
    --output_file ../data/${DATASET_NAME}/${DATASET_NAME}_schema_${SCHEMA_TYPE}_${SPLIT}.csv \
    --schema_name ${DATASET_NAME}_schema_${SCHEMA_TYPE} \
    --prompt_file ./prompts/${DATASET_NAME}_schema_extraction_prompt.txt \
    --text_column ${TEXT_COLUMN}

echo "inference finished"