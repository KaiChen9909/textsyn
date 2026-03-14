#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_model_merge          # 任务名称
#SBATCH --account=CIS260108-ai           # 你的项目账户
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00                # 运行时长上限 (例如 24 小时)
#SBATCH --output=logs/%j_merge_stdout.txt    # 标准输出

# ==========================================
# 环境准备
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load anaconda
source activate syn

export HF_HOME="/anvil/scratch/x-kchen28/.cache/huggingface"
cd $SLURM_SUBMIT_DIR

# ==========================================
# 运行你的 Bash 脚本任务
# ==========================================
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# ==========================================
# 参数解析: sbatch merge_model.sh <dataset_name> <algo> [eps] [use_dp] [epoch_id]
#   dataset_name: biorxiv
#   algo:         condgen / condgen_filter
#   eps:          epsilon value (default: 4.0)
#   use_dp:       1 for DP, 0 for non-DP (default: 1)
#   epoch_id:     epoch checkpoint to merge (default: 79)
# ==========================================
DATASET_NAME=${1:? "Missing argument: dataset name"}
ALGO=${2:?         "Missing argument: algorithm name"}
EPS=${3:-4.0}
USE_DP=${4:-1}
GPU_NUM=${5:-4}
EPOCH_ID=${6:-79}
MODEL_NAME=${7:-gemma}

# param for pe
PREV_N_GEN=${8:-20000}
L=${9:-4}

if [ "${MODEL_NAME}" = "gemma" ]; then
  MODEL_PT="gemma-3-1b-pt"
  MODEL_HF="google/gemma-3-1b-pt"
  MODEL_STR="gemma-3-1b"
elif [ "${MODEL_NAME}" = "qwen" ]; then
  MODEL_PT="Qwen2.5-1.5B-Instruct"
  MODEL_HF="Qwen/Qwen2.5-1.5B-Instruct"
  MODEL_STR="qwen2.5-1.5b"
else 
  echo "Unknown model" >&2
  exit 1
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

  if [ "${ALGO}" = "condgen" ]; then
    MAX_INST_LEN="300"; 
    LR_SCHEDULER="constant"; 
    TRAIN_DATASET="${DATASET_NAME}_${ALGO}"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then 
        NP="4.3"
      elif [ "${EPS}" = "1.0" ]; then 
        NP="13.8"
      else
        echo "Error: NP not defined for eps=${EPS}." >&2; exit 1;
      fi
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
    else
      EPS="-1.0"; 
      DELTA="0.1"; 
      CLIP="-1.0"; 
      NP="-1"
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
    fi

    DIR_NAME="${JOB_SESS}_${TRAIN_DATASET}_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"

  elif [ "${ALGO}" = "dpft" ]; then
    MAX_INST_LEN="32";
    LR_SCHEDULER="cosine";
    TRAIN_DATASET="${DATASET_NAME}"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        # NP="3.013"
        NP="3.15"
      elif [ "${EPS}" = "1.0" ]; then 
        NP="10.26"
      else
        echo "Error: NP not defined for eps=${EPS}." >&2; exit 1;
      fi
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
    else
      EPS="-1.0";
      DELTA="0.1";
      CLIP="-1.0";
      NP="-1"
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
    fi

    DIR_NAME="${JOB_SESS}_${TRAIN_DATASET}_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"

  elif [ "${ALGO}" = "condgen_filter" ]; then
    # condgen_filter uses different BS/STEP from default
    BS="1024";
    STEP="1280";
    MAX_INST_LEN="300";
    LR_SCHEDULER="cosine";
    TRAIN_DATASET="${DATASET_NAME}_${ALGO}"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        PREV_NP="4.3"
        RHO_FILTER="0.03"
      elif [ "${EPS}" = "1.0" ]; then 
        NP="13.8"
        RHO_FILTER="0.0025"
      else
        echo "Error: NP not defined for eps=${EPS}." >&2; exit 1;
      fi
      # Training data generated with DP (preveps from EPS parameter)
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_preveps${EPS}-np${PREV_NP}-rho${RHO_FILTER}-n${PREV_N_GEN}-L${L}_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
    else
      # Training data generated without DP (aligns with gen_filter.sh USE_DP=0)
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_preveps-1.0-np-1-rho0.0-n${PREV_N_GEN}-L${L}_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
    fi

    # Training itself is always non-DP, so use non-DP parameters
    EPS="-1.0";
    DELTA="0.1";
    CLIP="-1.0";
    NP="-1"
    DIR_NAME="${JOB_SESS}_${TRAIN_DATASET}_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"

  else
    echo "Error: Unknown algo '${ALGO}'. Supported: noexample, noexample_4attr, noexample_8outof24attr, noexample_16attr, noexample_24attr, example50, dpft, condgen_filter" >&2
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
echo "Merging epoch: ${EPOCH_ID}"
bash scripts/gen/run_merge.sh "${DIR_NAME}" "${EPOCH_ID}" "${MODEL_HF}"

echo "Job finished at: $(date)"