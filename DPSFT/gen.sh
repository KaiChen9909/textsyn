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
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_gen_stdout.txt
#SBATCH --error=logs/%j_gen_stderr.txt

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

# ==========================================
# 参数解析: sbatch gen.sh <dataset_name> <algo> [eps]
#   dataset_name: biorxiv
#   algo:         noexample / example50 / dpft
#   eps:          epsilon value (default: 4.0)
# ==========================================
DATASET_NAME=${1:? "Usage: sbatch gen.sh <dataset_name> <algo> [eps]"}
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
    NP="4.3"
    RHO="0.18"
    PROMPT_FILE="../AIM/results/synthetic_biorxiv_noexample_et_5k_rho-${RHO}_iter-2000.csv"
    DIR_NAME="${MODEL_STR}_biorxiv_noexample_bs-${BS}_step-${STEP}_lr-${LR}-constant_seed-42_biorxiv_noexample_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq300-512_epoch${EPOCH}_lr${LR_VAL}_clip1.0_np${NP}_gpus${GPUS}"
  
  elif [ "${ALGO}" = "example50" ]; then
    NP="4.3"
    RHO="0.18"
    PROMPT_FILE="../AIM/results/synthetic_biorxiv_example50_et_5k_rho-${RHO}_iter-2000.csv"
    DIR_NAME="${MODEL_STR}_biorxiv_example50_bs-${BS}_step-${STEP}_lr-${LR}-constant_seed-42_biorxiv_example50_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq300-512_epoch${EPOCH}_lr${LR_VAL}_clip1.0_np${NP}_gpus${GPUS}"
  
  elif [ "${ALGO}" = "dpft" ]; then
    NP="3.013"
    DIR_NAME="${MODEL_STR}_biorxiv_dpft_bs-${BS}_step-${STEP}_lr-${LR}-cosine_seed-42_biorxiv_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq32-512_epoch${EPOCH}_lr${LR_VAL}_clip1.0_np${NP}_gpus${GPUS}"
  else
    echo "Error: Unknown algo '${ALGO}'. Supported: noexample, example50, dpft" >&2
    exit 1
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

OUTPUT_BASE="results/outputs"
MODEL_DIR="${OUTPUT_BASE}/${DIR_NAME}/model_epoch${EPOCH_ID}"
if [ ! -d "${MODEL_DIR}" ]; then
  echo "Error: Merged model not found: ${MODEL_DIR}" >&2
  exit 1
fi

echo "Using model: ${MODEL_DIR}"


if [ "${ALGO}" = "dpft" ]; then
  bash scripts/gen/run_biorxiv_gen_baseline.sh "${DIR_NAME}" "${EPOCH_ID}" "${EPS}" "${NP}" "${LR}" 512
else
  bash scripts/gen/run_biorxiv_gen_features.sh "${DIR_NAME}" "${EPOCH_ID}" "${EPS}" "${NP}" "${LR}" 300 512 "${DATASET_NAME}_${ALGO}" "${PROMPT_FILE}"
fi

echo "Job finished at: $(date)"
