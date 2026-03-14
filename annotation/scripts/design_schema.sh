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
#SBATCH -o logs/%j_design_schema_stdout.txt

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
ALGO=${2:-noexample}
NUM_FEATURES=${3:-8}

echo "designing schema ..."

python design_schema.py \
    --dataset_name ${DATASET_NAME} \
    --output_name ${DATASET_NAME}_schema_${ALGO} \
    --num_features ${NUM_FEATURES} 

echo "design finished"