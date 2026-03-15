#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_gen_filter
#SBATCH --account=CIS260108-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_gen_filter_stdout.txt

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
ALGO=${2:? "Missing argument: algorithm name"}
EPS=${3:-4.0}
USE_DP=${4:-1}
L=${5:-4}
RHO_FILTER=${6:-0.03}
RHO_PREV=${7:-0.15}
GPU_NUM=${8:-4}
EPOCH_ID=${9:-79}
MODEL=${10:-gemma}
N_GEN=${11:-5000}
ROUND=${12:-1}
VARIATION_MODEL_TYPE=${13:-qwen}
EVALUATE=${14:-1}

if [ "${MODEL}" = "gemma" ]; then
  MODEL_STR="gemma-3-1b"
  MODEL_PT="gemma-3-1b-pt"
elif [ "${MODEL}" = "qwen" ]; then
  MODEL_STR="qwen2.5-1.5b"
  MODEL_PT="Qwen2.5-1.5B-Instruct"
fi

# Set variation model path based on type
if [ "${VARIATION_MODEL_TYPE}" = "qwen" ]; then
  VARIATION_MODEL="Qwen/Qwen2.5-7B-Instruct"
elif [ "${VARIATION_MODEL_TYPE}" = "qwen-3b" ]; then
  VARIATION_MODEL="Qwen/Qwen2.5-3B-Instruct"
elif [ "${VARIATION_MODEL_TYPE}" = "qwen-1.5b" ]; then
  VARIATION_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
elif [ "${VARIATION_MODEL_TYPE}" = "gemma" ]; then
  VARIATION_MODEL="google/gemma-2-9b-it"
else
  # If it's a full path, use it directly
  VARIATION_MODEL="${VARIATION_MODEL_TYPE}"
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

  if [ "${ALGO}" = "condgen_filter" ]; then
    if [ "${EPS}" = "4.0" ]; then
      NP="4.3"
    elif [ "${EPS}" = "1.0" ]; then
      NP="13.8"
    else
      echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
      exit 1
    fi

    MAX_INST_LEN="300"
    LR_SCHEDULER="constant"
    PROMPT_STR="${DATASET_NAME}_condgen"
    PROMPT_FILE="../AIM/results/synthetic_${DATASET_NAME}_condgen_et_5k_rho-${RHO_PREV}_iter-2000.csv"
    REAL_DATA_PATH="../data/biorxiv/train.csv"

    if [ "${USE_DP}" = "1" ]; then
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_condgen_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
      MODEL_DIR="${JOB_SESS}_biorxiv_condgen_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"
    else
      EPS="-1.0"
      DELTA="0.1"
      CLIP="-1.0"
      NP="-1"
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_condgen_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
      MODEL_DIR="${JOB_SESS}_biorxiv_condgen_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"
    fi

  elif [ "${ALGO}" = "dpft_filter" ]; then
    if [ "${EPS}" = "4.0" ]; then
      NP="3.15"
    elif [ "${EPS}" = "1.0" ]; then
      NP="27.6"
    else
      echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
      exit 1
    fi

    MAX_INST_LEN="32" 
    LR_SCHEDULER="cosine" 
    PROMPT_STR=""  # No prompt needed for dpft_filter
    PROMPT_FILE=""  # No prompt file needed
    REAL_DATA_PATH="../data/biorxiv/train.csv"

    if [ "${USE_DP}" = "1" ]; then
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_dpft_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
      MODEL_DIR="${JOB_SESS}_${DATASET_NAME}_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"
    else
      EPS="-1.0"
      DELTA="0.1"
      CLIP="-1.0"
      NP="-1"
      JOB_SESS="${MODEL_STR}_${DATASET_NAME}_dpft_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER}_seed-42"
      MODEL_DIR="${JOB_SESS}_${DATASET_NAME}_noredacted_model${MODEL_PT}_eps${EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${NP}_gpus${GPUS}"
    fi

  else
    echo "Error: Unknown algorithm '${ALGO}'. Supported: condgen_filter, dpft_filter" >&2
    exit 1
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi


OUTPUT_STR="${DATASET_NAME}_${ALGO}"
MODEL_PATH="results/outputs/${MODEL_DIR}/model_epoch${EPOCH_ID}"

echo "Using model dir: ${MODEL_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Output str: ${OUTPUT_STR}"
echo "Epoch: ${EPOCH_ID}"
echo "L (oversampling): ${L}"
echo "N_GEN: ${N_GEN}"
echo "ROUND: ${ROUND}"
echo "VARIATION_MODEL_TYPE: ${VARIATION_MODEL_TYPE}"
echo "VARIATION_MODEL: ${VARIATION_MODEL}"
echo "EVALUATE: ${EVALUATE}"

set -x

if [ "${ALGO}" = "condgen_filter" ]; then
  # Build evaluate flag (default: enabled)
  EVAL_FLAG="--evaluate"
  if [ "${EVALUATE}" = "0" ] || [ "${EVALUATE}" = "false" ]; then
    EVAL_FLAG=""
  fi

  # Build output filename based on round
  if [ "${ROUND}" = "1" ]; then
    OUT_FILENAME="generated_${OUTPUT_STR}_rho-${RHO_FILTER}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}.jsonl"
  else
    OUT_FILENAME="generated_${OUTPUT_STR}_rho-${RHO_FILTER}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}_round-${ROUND}.jsonl"
  fi

  python generation_condgen_filter.py \
      -m ${MODEL_PATH} \
      -pl ${MAX_INST_LEN} -sl ${SEQLEN} -d 0 \
      -o ${OUTPUT_STR} -ps ${PROMPT_STR} \
      -out ${OUT_FILENAME} \
      -n_gen ${N_GEN} -L ${L} -bs 512 -tp 0.95 \
      -pf ${PROMPT_FILE} \
      -rd ${REAL_DATA_PATH} -tc abstract \
      -rho ${RHO_FILTER} \
      --round ${ROUND} \
      --variation_model ${VARIATION_MODEL} \
      ${EVAL_FLAG}

elif [ "${ALGO}" = "dpft_filter" ]; then
  python generation_gen_filter.py \
      -m ${MODEL_PATH} \
      -sl ${SEQLEN} -d 0 \
      -o ${OUTPUT_STR} -ps ${DATASET_NAME} \
      -out generated_${OUTPUT_STR}_rho-${RHO_FILTER}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}.jsonl \
      -n_gen ${N_GEN} -L ${L} -bs 512 -tp 0.95 \
      -rd ${REAL_DATA_PATH} -tc abstract \
      -rho ${RHO_FILTER}
fi

echo "Job finished at: $(date)"
