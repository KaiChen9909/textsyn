#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH -J biorxiv_gen
#SBATCH -A dplab
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o logs/%j_gen_stdout.txt

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
EPOCH_ID=${6:-79}
MODEL=${7:-gemma}
N_GEN=${8:-5000}
PREV_N_GEN=${9:-20000}
L=${10:-4}

if [ "${MODEL}" = "gemma" ]; then 
  MODEL_STR="gemma-3-1b"
  MODEL_PT="gemma-3-1b-pt"
elif [ "${MODEL}" = "qwen" ]; then
  MODEL_STR="qwen2.5-1.5b"
  MODEL_PT="Qwen2.5-1.5B-Instruct"
fi


if [ "${DATASET_NAME}" = "biorxiv" ]; then
  DELTA="3.38e-06"
  BS="2048"
  STEP="1120"
  LR="1e-3"
  EPOCH="3"
  LR_VAL="0.001"
  GPUS="${GPU_NUM}"
  SEQLEN="512"
  CLIP="1.0"
  RHO_FILTER=""

  if [[ "${ALGO}" = "condgen" ]]; then
    MAX_INST_LEN="300"
    LR_SCHEDULER="constant"
    TRAIN_DATASET="${DATASET_NAME}_${ALGO}"

    if [ "${EPS}" = "4.0" ]; then
      NP="4.3"
      RHO="0.18"
    elif [ "${EPS}" = "1.0" ]; then
      NP="13.8"
      RHO="0.015"
    else
      echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
      exit 1
    fi

    PROMPT_STR="${DATASET_NAME}_${ALGO}"
    PROMPT_FILE="../AIM/results/synthetic_${DATASET_NAME}_${ALGO}_et_5k_rho-${RHO}_iter-2000.csv"

  elif [[ "${ALGO}" = "condgen_filter" ]]; then
    BS="1024"
    STEP="1280"
    MAX_INST_LEN="300"
    LR_SCHEDULER="cosine"
    TRAIN_DATASET="${DATASET_NAME}_${ALGO}"

    if [ "${EPS}" = "4.0" ]; then
      NP="4.3"
      RHO="0.15"
      RHO_FILTER="0.03"
    elif [ "${EPS}" = "1.0" ]; then
      NP="13.8"
      RHO="0.0125"
      RHO_FILTER="0.0025"
    else
      echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
      exit 1
    fi

    PROMPT_STR="${DATASET_NAME}_condgen"
    PROMPT_FILE="../AIM/results/synthetic_${DATASET_NAME}_condgen_et_5k_rho-${RHO}_iter-2000.csv"

  elif [ "${ALGO}" = "dpft" ]; then
    MAX_INST_LEN="32"
    LR_SCHEDULER="cosine"
    TRAIN_DATASET="${DATASET_NAME}"

    if [ "${EPS}" = "4.0" ]; then
      NP="3.013"
    elif [ "${EPS}" = "1.0" ]; then
      NP="10.26"
    else
      echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
      exit 1
    fi

  else
    echo "Error: Unknown algo '${ALGO}'. Supported: noexample, noexample_4attr, noexample_8outof24attr, noexample_16attr, noexample_24attr, example50, dpft, noft, noft_noexample" >&2
    exit 1
  fi

  # {job_sess}_{dataset_name}_noredacted_model{model}_eps{eps}_delta{delta}_bs{bs}_maxseq{inst_len-seq_len}_epoch{epoch}_lr{lr}_clip{clip}_np{np}_gpus{gpus}
  if [ "${USE_DP}" = "1" ]; then
    if [ "${ALGO}" = "condgen_filter" ]; then
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_preveps${EPS}-np${NP}-rho${RHO_FILTER}-n${PREV_N_GEN}-L${L}_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
      DIR_NAME="${JOB_SESS}_${TRAIN_DATASET}_noredacted_model${MODEL_PT}_eps-1.0_delta0.1_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip-1.0_np-1_gpus${GPUS}"
    else
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
      DIR_NAME="${JOB_SESS}_${TRAIN_DATASET}_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"
    fi
  else
    # Non-DP
    EPS="-1.0"
    DELTA="0.1"
    CLIP="-1.0"
    NP="-1"
    if [ "${ALGO}" = "condgen_filter" ]; then
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_preveps${EPS}-np${NP}-rho0.0-n${PREV_N_GEN}-L${L}_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
      DIR_NAME="${JOB_SESS}_${TRAIN_DATASET}_noredacted_model${MODEL_PT}_eps-1.0_delta0.1_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip-1.0_np-1_gpus${GPUS}"
    else
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
      DIR_NAME="${JOB_SESS}_${TRAIN_DATASET}_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"
    fi
  fi

else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

echo "Using directory: ${DIR_NAME}"
echo "Epoch: ${EPOCH_ID}"

if [ "${ALGO}" = "dpft" ]; then
  bash scripts/gen/run_biorxiv_gen_baseline.sh "${DIR_NAME}" "${EPOCH_ID}" "${EPS}" "${NP}" "${LR}" "${STEP}" "${RHO_FILTER}" "${SEQLEN}" "${DATASET_NAME}_${ALGO}"
elif [ "${ALGO}" = "condgen_filter" ]; then
  bash scripts/gen/run_biorxiv_gen_features.sh "${DIR_NAME}" "${EPOCH_ID}" "${EPS}" "${NP}" "${LR}" "${STEP}" "${RHO_FILTER}" "${MAX_INST_LEN}" "${SEQLEN}" "${DATASET_NAME}_${ALGO}" "${PROMPT_STR}" "${PROMPT_FILE}" "${N_GEN}" "${MODEL_PT}" "${PREV_N_GEN}" "${L}" "${SCHEMA_COLUMN}"
else
  bash scripts/gen/run_biorxiv_gen_features.sh "${DIR_NAME}" "${EPOCH_ID}" "${EPS}" "${NP}" "${LR}" "${STEP}" "${RHO_FILTER}" "${MAX_INST_LEN}" "${SEQLEN}" "${DATASET_NAME}_${ALGO}" "${PROMPT_STR}" "${PROMPT_FILE}" "${N_GEN}" "${MODEL_PT}"
fi

echo "Job finished at: $(date)"
