#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_model_merge          # 任务名称
#SBATCH --account=NAIRR250463-ai           # 你的项目账户
#SBATCH --partition=ai                 
#SBATCH --nodes=1                      
#SBATCH --gpus-per-node=1             
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=16  
#SBATCH --mem=96G         
#SBATCH --time=24:00:00                # 运行时长上限 (例如 24 小时)
#SBATCH --output=logs/%j_merge_stdout.txt    # 标准输出
#SBATCH --error=logs/%j_merge_stderr.txt     # 错误输出

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


DATASET_NAME=${1:? "Usage: sbatch merge_model.sh <dataset_name> <algo> <epoch_id> [eps]"}
ALGO=${2:?         "Missing argument: algo (noexample, example50, dpft)"}
EPS=${3:-4.0}


if [ "${DATASET_NAME}" = "biorxiv" ]; then
  MODEL_STR="gemma-3-1b"
  MODEL_PT="gemma-3-1b-pt"
  DELTA="3.38e-06"
  BS="2048"
  STEP="1120"
  LR="1e-3"
  EPOCH="3"
  LR_VAL="0.001"
  GPUS="2"
  EPOCH_ID="79"

  if [ "${ALGO}" = "noexample" ]; then
    DIR_NAME="${MODEL_STR}_biorxiv_noexample_bs-${BS}_step-${STEP}_lr-${LR}-constant_seed-42_biorxiv_noexample_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq300-512_epoch${EPOCH}_lr${LR_VAL}_clip1.0_np4.3_gpus${GPUS}"
  elif [ "${ALGO}" = "example50" ]; then
    DIR_NAME="${MODEL_STR}_biorxiv_example50_bs-${BS}_step-${STEP}_lr-${LR}-constant_seed-42_biorxiv_example50_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq300-512_epoch${EPOCH}_lr${LR_VAL}_clip1.0_np4.3_gpus${GPUS}"
  elif [ "${ALGO}" = "dpft" ]; then
    DIR_NAME="${MODEL_STR}_biorxiv_dpft_bs-${BS}_step-${STEP}_lr-${LR}-cosine_seed-42_biorxiv_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq32-512_epoch${EPOCH}_lr${LR_VAL}_clip1.0_np3.013_gpus${GPUS}"
  else
    echo "Error: Unknown algo '${ALGO}'. Supported: noexample, example50, dpft" >&2
    exit 1
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

OUTPUT_BASE="results/outputs"
if [ ! -d "${OUTPUT_BASE}/${DIR_NAME}" ]; then
  echo "Error: Directory not found: ${OUTPUT_BASE}/${DIR_NAME}" >&2
  exit 1
fi

echo "Matched directory: ${DIR_NAME}"
bash scripts/run_merge.sh "${DIR_NAME}" "${EPOCH_ID}"

echo "Job finished at: $(date)"