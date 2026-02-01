#!/bin/bash

# ==========================================
# SLURM 资源配置 (针对 2 块 H100)
# ==========================================
#SBATCH --job-name=biorxiv_dpft          # 任务名称
#SBATCH --account=NAIRR250463-ai       # 你的项目账户
#SBATCH --partition=ai                 # 必须选 ai 分区以使用 H100
#SBATCH --nodes=1                      # 使用 1 个计算节点
#SBATCH --gpus-per-node=2              # 关键：申请 2 块 GPU
#SBATCH --ntasks-per-node=1            # 脚本作为单个任务运行
#SBATCH --cpus-per-task=32             # 2 块 GPU 建议配 32 个 CPU 核心 (16x2)
#SBATCH --mem=128G                     # 建议增加内存以匹配多显卡任务
#SBATCH --time=24:00:00                # 运行时长上限 (例如 24 小时)
#SBATCH --output=logs/%j_dpft_stdout.txt    # 标准输出
#SBATCH --error=logs/%j_dpft_stderr.txt     # 错误输出

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

bash scripts/run_biorxiv_ft_dp.sh 2048 1120 1e-3 29500 4.0 3.013 512 4 2 biorxiv cosine

echo "Job finished at: $(date)"
