#!/bin/bash

# ==========================================
# SLURM 资源配置 (针对 2 块 H100)
# ==========================================
#SBATCH -J biorxiv_train
#SBATCH -A dplab
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -C a100_80gb
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 27:00:00
#SBATCH -o logs/%j_train_stdout.txt

# ==========================================
# 环境准备
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load uv
source /scratch/pkq2ps/envs/syn/bin/activate

export HF_HOME="/anvil/scratch/x-kchen28/.cache/huggingface"
cd $SLURM_SUBMIT_DIR

# ==========================================
# 运行你的 Bash 脚本任务
# ==========================================
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"


DATASET_NAME=${1:? "Missing argument: dataset name"}
ALGO=${2:?         "Missing argument: algorithm name"}
EPS=${3:-4.0}
USE_DP=${4:-1}
GPU_NUM=${5:-4}
MODEL_NAME=${6:-gemma}
PORT_ID=${7:-29500}

if [ "${MODEL_NAME}" = "gemma" ]; then 
  GEN_MODEL="google/gemma-3-1b-pt"
  MODEL_STR="gemma-3-1b"
else 
  echo "Unknown model" >&2
  exit 1
fi

if [ "${DATASET_NAME}" = "biorxiv" ]; then
  BS="2048"
  STEP="1120"
  LR="1e-3"
  PORT="${PORT_ID}"
  SEQLEN="512"
  GPUS="${GPU_NUM}"
  DEVICE_BS="4"
  DELTA="3.38e-6"

  if [ "${EPS}" = "4.0" ]; then
    if [[ "${ALGO}" = "condgen"* ]]; then
      NP="4.3"
      MAX_INST_LEN=300
    elif [ "${ALGO}" = "dpft" ]; then
      NP="3.013"
      # NP="3.15" # for dpft_filter
    else
      echo "Error: Unknown algo '${ALGO}'" >&2
      exit 1
    fi
  elif [ "${EPS}" = "1.0" ]; then 
    if [[ "${ALGO}" = "condgen"* ]]; then
      NP="13.8" 
      MAX_INST_LEN=300
    elif [ "${ALGO}" = "dpft" ]; then
      NP="10.26"
      # NP="10.8" # for dpft_filter
    else
      echo "Error: Unknown algo '${ALGO}'" >&2
      exit 1
    fi
  fi
elif [ "${DATASET_NAME}" = "openreview" ]; then
  BS="2048"
  STEP="320"
  LR="1e-3"
  PORT="${PORT_ID}"
  SEQLEN="1024"
  GPUS="${GPU_NUM}"
  DEVICE_BS="2"
  DELTA="1.32e-5"

  if [ "${EPS}" = "4.0" ]; then
    if [[ "${ALGO}" = "condgen"* ]]; then
      NP="7.0"
      MAX_INST_LEN=300
    elif [ "${ALGO}" = "dpft" ]; then
      NP="5.11"
      # NP="5.4" # for dpft_filter
    else
      echo "Error: Unknown algo '${ALGO}'" >&2
      exit 1
    fi
  elif [ "${EPS}" = "1.0" ]; then
    if [[ "${ALGO}" = "condgen"* ]]; then
      NP="24.1"
      MAX_INST_LEN=300
    elif [ "${ALGO}" = "dpft" ]; then
      NP="17.48"
    else
      echo "Error: Unknown algo '${ALGO}'" >&2
      exit 1
    fi
  fi
elif [ "${DATASET_NAME}" = "pmc" ]; then
  BS=4096
  STEP=960
  LR="1e-3"
  PORT="${PORT_ID}"
  SEQLEN="1024"
  GPUS="${GPU_NUM}"
  DEVICE_BS="2"
  DELTA="1.85e-6"

  if [ "${EPS}" = "4.0" ]; then
    if [[ "${ALGO}" = "condgen"* ]]; then
      NP="4.8"
      MAX_INST_LEN=300
    elif [ "${ALGO}" = "dpft" ]; then
      NP="3.28"
      # NP="3.5"
    else
      echo "Error: Unknown algo '${ALGO}'" >&2
      exit 1
    fi
  elif [ "${EPS}" = "1.0" ]; then
    if [[ "${ALGO}" = "condgen"* ]]; then
      NP="17.4"
      MAX_INST_LEN=300
    elif [ "${ALGO}" = "dpft" ]; then
      NP="11.26"
      # NP="11.9"
    else
      echo "Error: Unknown algo '${ALGO}'" >&2
      exit 1
    fi
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'" >&2
  exit 1
fi


if [ "${USE_DP}" = "1" ]; then
  if [ "${ALGO}" = "dpft" ]; then
    bash scripts/train/run_biorxiv_ft_dp.sh ${BS} ${STEP} ${LR} ${PORT} ${EPS} ${NP} ${DELTA} ${SEQLEN} ${DEVICE_BS} ${GPUS} ${DATASET_NAME} cosine ${GEN_MODEL} ${MODEL_STR}
  else
    bash scripts/train/run_biorxiv-condgen_ft_dp.sh ${BS} ${STEP} ${LR} ${PORT} ${EPS} ${NP} ${DELTA} ${SEQLEN} ${DEVICE_BS} ${GPUS} ${DATASET_NAME}_${ALGO} ${MAX_INST_LEN} constant ${GEN_MODEL} ${MODEL_STR}
  fi
else
  if [ "${ALGO}" = "dpft" ]; then
    bash scripts/train/run_biorxiv_ft_nondp.sh ${BS} ${STEP} ${LR} ${PORT} ${SEQLEN} ${DEVICE_BS} ${GPUS} ${DATASET_NAME} cosine ${GEN_MODEL} ${MODEL_STR}
  else
    bash scripts/train/run_biorxiv-condgen_ft_nondp.sh ${BS} ${STEP} ${LR} ${PORT} ${SEQLEN} ${DEVICE_BS} ${GPUS} ${DATASET_NAME}_${ALGO} ${MAX_INST_LEN} constant ${GEN_MODEL} ${MODEL_STR}
  fi
fi


echo "Job finished at: $(date)"
