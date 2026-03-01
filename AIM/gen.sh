#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_noexample_aim       # 任务名称
#SBATCH --account=NAIRR250463-ai       # 你的项目账户
#SBATCH --partition=ai                 # 必须选 ai 分区以使用 H100
#SBATCH --nodes=1                      # 使用 1 个计算节点
#SBATCH --gpus-per-node=1              # 关键：申请 1 块 GPU
#SBATCH --ntasks-per-node=1            # 脚本作为单个任务运行
#SBATCH --cpus-per-task=16             
#SBATCH --mem=32G                     # 建议增加内存以匹配多显卡任务
#SBATCH --time=12:00:00                # 运行时长上限 (例如 24 小时)
#SBATCH --output=logs/%j_noexample_aim_stdout.txt    # 标准输出

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