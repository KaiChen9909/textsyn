#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_example50_aim       # 任务名称
#SBATCH --account=NAIRR250463-ai       # 你的项目账户
#SBATCH --partition=ai                 # 必须选 ai 分区以使用 H100
#SBATCH --nodes=1                      # 使用 1 个计算节点
#SBATCH --gpus-per-node=1              # 关键：申请 1 块 GPU
#SBATCH --ntasks-per-node=1            # 脚本作为单个任务运行
#SBATCH --cpus-per-task=16             
#SBATCH --mem=64G                     # 建议增加内存以匹配多显卡任务
#SBATCH --time=12:00:00                # 运行时长上限 (例如 24 小时)
#SBATCH --output=logs/%j_example50_aim_stdout.txt    # 标准输出
#SBATCH --error=logs/%j_example50_aim_stderr.txt     # 错误输出

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

python main.py --rho 0.18 \
        --data_path ../data/biorxiv/clean_biorxiv_schema_example50_train.csv \
        --schema_path ../annotation/schema/biorxiv_schema_example50_test.txt \
        --output_name synthetic_biorxiv_example50

echo "Job finished at: $(date)"