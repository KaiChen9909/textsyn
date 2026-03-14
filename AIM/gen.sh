#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH -J biorxiv_noexample_aim
#SBATCH -A dplab
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%j_noexample_aim_stdout.txt

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
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

DATASET_NAME=${1:? "Missing argument: dataset name"}
ALGO=${2:?         "Missing argument: algorithm name"}
RHO=${3:-0.18}

# check is GPU is available
python -c "import jax; print('JAX devices:', jax.devices())"

python main.py --rho "${RHO}"\
        --data_path "../data/${DATASET_NAME}/clean_${DATASET_NAME}_schema_${ALGO}_train.csv" \
        --schema_path "../annotation/schema/${DATASET_NAME}_schema_${ALGO}.txt" \
        --output_name "synthetic_${DATASET_NAME}_${ALGO}"

echo "Job finished at: $(date)"