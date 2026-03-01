#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_eval_fid
#SBATCH --account=NAIRR250463-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_eval_fid_stdout.txt

# ==========================================
# 环境准备
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load anaconda
source activate syn

# ==========================================
# 运行任务
# ==========================================
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"


DATASET_NAME=${1:? "Missing argument: dataset name"}
ALGO=${2:?         "Missing argument: algorithm name"}
EPS=${3:-4.0}
USE_DP=${4:-1}
GPU_NUM=${5:-2}
SAVE_PATH=${6:-"none"}
MODEL=${7:-gemma}

# Resolve model strings (same convention as eval_mauve.sh)
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

  N_GEN=5000

  if [ "${ALGO}" = "noexample" ] || [ "${ALGO}" = "noexample_4attr" ] || [ "${ALGO}" = "noexample_8outof24attr" ] || [ "${ALGO}" = "example50" ] || [ "${ALGO}" = "noft_noexample" ] || [ "${ALGO}" = "noexample_pretrain" ]; then
    MAX_INST_LEN="300"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="jsonl"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        NP="4.3"
      else
        echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
        exit 1
      fi
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    else
      EPS="-1.0"
      NP="-1"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    fi

  elif [ "${ALGO}" = "condgen_pretrain" ]; then
    MAX_INST_LEN="300"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="jsonl"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        NP="4.75"
      else
        echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
        exit 1
      fi
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    else
      EPS="-1.0"
      NP="-1"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    fi

  elif [ "${ALGO}" = "noexample_16attr" ] || [ "${ALGO}" = "noexample_24attr" ]; then
    MAX_INST_LEN="600"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="jsonl"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        NP="4.3"
      else
        echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
        exit 1
      fi
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    else
      EPS="-1.0"
      NP="-1"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    fi

  elif [ "${ALGO}" = "dpft" ] || [ "${ALGO}" = "noft" ]; then
    MAX_INST_LEN="32"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="csv"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        NP="3.013"
      else
        echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
        exit 1
      fi
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-nm-${NP}_lr-${LR}_seqlen-${SEQLEN}_n-${N_GEN}"
    else
      EPS="-1.0"
      NP="-1"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-nm-${NP}_lr-${LR}_seqlen-${SEQLEN}_n-${N_GEN}"
    fi

  else
    echo "Error: Unknown algo '${ALGO}'. Supported: noexample, noexample_4attr, noexample_8outof24attr, noexample_16attr, noexample_24attr, example50, dpft, noft, noft_noexample, condgen_pretrain, noexample_pretrain" >&2
    exit 1
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

GPU_DEVICE=0
EMBED_FILE="${FILE_STEM}_specter2_len-512_embeddings_rerun.npy"

echo "Mode: ${MODE}"
echo "File stem: ${FILE_STEM}"
echo "Embedding file: ${EMBED_FILE}"

# Reuse embedding if already computed by eval_mauve.sh; otherwise compute it.
MAUVE_EMBED_DIR="../mauve/embeddings"
LOCAL_EMBED_DIR="embeddings"

if [ -f "${MAUVE_EMBED_DIR}/${EMBED_FILE}" ]; then
  echo "Found existing MAUVE embedding, reusing: ${MAUVE_EMBED_DIR}/${EMBED_FILE}"
  Q_EMBED_PATH="${MAUVE_EMBED_DIR}/${EMBED_FILE}"
elif [ -f "${LOCAL_EMBED_DIR}/${EMBED_FILE}" ]; then
  echo "Found existing local embedding: ${LOCAL_EMBED_DIR}/${EMBED_FILE}"
  Q_EMBED_PATH="${LOCAL_EMBED_DIR}/${EMBED_FILE}"
else
  echo "No existing embedding found, computing embeddings..."
  bash scripts/run_embed_biorxiv.sh "${FILE_STEM}" "${GPU_DEVICE}" "${MODE}" "${FILE_TYPE}"
  Q_EMBED_PATH="${LOCAL_EMBED_DIR}/${EMBED_FILE}"
fi

echo "Computing FID score..."
bash scripts/run_compute_fid_biorxiv.sh "${Q_EMBED_PATH}" "${SAVE_PATH}"

echo "Job finished at: $(date)"
