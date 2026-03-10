#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_eval_mauve
#SBATCH --account=CIS260108-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_eval_mauve_stdout.txt

# ==========================================
# 环境准备
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load anaconda
source activate syn

# cd $SLURM_SUBMIT_DIR

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
GPU_NUM=${5:-2}
SAVE_PATH=${6:-"none"}
MODEL=${7:-gemma}
DETAILED_ANALYSIS=${8:-true}  # detailed MAUVE analysis
RHO_FILTER=${9:-0.03}
DATA_TYPE=${10:-final}  # condgen_filter: "final" (from gen.sh) or "intermediate" (from gen_filter.sh)
STEP=${11:-1120}
PREV_N_GEN=${12:-5000}
L=${13:-4}

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
  LR="1e-3"
  EPOCH="3"
  LR_VAL="0.001"
  GPUS="${GPU_NUM}"
  SEQLEN="512"
  CLIP="1.0"

  N_GEN=5000

  if [ "${ALGO}" = "condgen_filter" ]; then
    MODE="${DATASET_NAME}_${ALGO}"

    if [ "${DATA_TYPE}" = "intermediate" ]; then
      # Intermediate filtered data from gen_filter.sh
      MAX_INST_LEN="300"
      FILE_TYPE="jsonl"

      if [ "${USE_DP}" = "1" ]; then
        if [ "${EPS}" = "4.0" ]; then
          NP="4.3"
        elif [ "${EPS}" = "1.0" ]; then
          NP="13.8"
        else
          echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
          exit 1
        fi
        FILE_STEM="${DATASET_NAME}_${ALGO}_rho-${RHO_FILTER}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}"
      else
        EPS="-1.0"
        NP="-1"
        FILE_STEM="${DATASET_NAME}_${ALGO}_rho-${RHO_FILTER}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}"
      fi

    elif [ "${DATA_TYPE}" = "final" ]; then
      # Final synthetic data from gen.sh
      MAX_INST_LEN="300"
      FILE_TYPE="jsonl"

      if [ "${USE_DP}" = "1" ]; then
        if [ "${EPS}" = "4.0" ]; then
          NP="4.3"
        elif [ "${EPS}" = "1.0" ]; then
          NP="13.8"
        else
          echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
          exit 1
        fi
        FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}-step-${STEP}_rho-${RHO_FILTER}_n${PREV_N_GEN}_L${L}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
      else
        EPS="-1.0"
        NP="-1"
        FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}-step-${STEP}_rho-${RHO_FILTER}_n${PREV_N_GEN}_L${L}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
      fi

    else
      echo "Error: Unknown DATA_TYPE '${DATA_TYPE}'. For condgen_filter, supported: intermediate, final" >&2
      exit 1
    fi

  elif [ "${ALGO}" = "condgen" ]; then
    MAX_INST_LEN="300"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="jsonl"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        NP="4.3"
      elif [ "${EPS}" = "1.0" ]; then
          NP="13.8"
      else
        echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
        exit 1
      fi
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}-step-${STEP}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    else
      # non-DP: overwrite EPS and NP
      EPS="-1.0"
      NP="-1"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}-step-${STEP}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    fi

  elif [ "${ALGO}" = "dpft_filter" ]; then
    MODE="${DATASET_NAME}_${ALGO}"

    if [ "${DATA_TYPE}" = "intermediate" ]; then
      # Intermediate filtered data from gen_filter.sh
      MAX_INST_LEN="32"
      FILE_TYPE="jsonl"

      if [ "${USE_DP}" = "1" ]; then
        if [ "${EPS}" = "4.0" ]; then
          NP="3.15"
        elif [ "${EPS}" = "1.0" ]; then
          NP="10.26"
        else
          echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
          exit 1
        fi
        FILE_STEM="${DATASET_NAME}_${ALGO}_rho-${RHO_FILTER}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}"
      else
        EPS="-1.0"
        NP="-1"
        FILE_STEM="${DATASET_NAME}_${ALGO}_rho-${RHO_FILTER}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}"
      fi

    else
      echo "Error: Unknown DATA_TYPE '${DATA_TYPE}'. For dpft_filter, only 'intermediate' is currently supported." >&2
      exit 1
    fi

  elif [ "${ALGO}" = "dpft" ]; then
    MAX_INST_LEN="32"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="csv"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        NP="3.013"
      elif [ "${EPS}" = "1.0" ]; then
          NP="10.26"
      else
        echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
        exit 1
      fi
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-nm-${NP}-step-${STEP}_lr-${LR}_seqlen-${SEQLEN}_n-${N_GEN}"
    else
      # non-DP: overwrite EPS and NP
      EPS="-1.0"
      NP="-1"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-nm-${NP}-step-${STEP}_lr-${LR}_seqlen-${SEQLEN}_n-${N_GEN}"
    fi

  else
    echo "Error: Unknown algo '${ALGO}'. Supported: condgen_filter, condgen, dpft_filter, dpft, noft" >&2
    exit 1
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

GPU_DEVICE=0
EMBED_FILE="${FILE_STEM}_specter2_len-512_embeddings_rerun.npy"

# Determine base path for data
BASE_PATH="synthetic"
if [ "${DATA_TYPE}" = "intermediate" ] && { [ "${ALGO}" = "condgen_filter" ] || [ "${ALGO}" = "dpft_filter" ]; }; then
  BASE_PATH="intermediate"
fi

echo "Mode: ${MODE}"
echo "File stem: ${FILE_STEM}"
echo "Base path: ${BASE_PATH}"
echo "Embedding file: generated_${FILE_STEM}.${FILE_TYPE} in generations_${MODE}/"

bash scripts/run_embed_biorxiv.sh "${FILE_STEM}" "${GPU_DEVICE}" "${MODE}" "${FILE_TYPE}" "" "generated_text" "${BASE_PATH}"

echo "Computing MAUVE score..."

# Determine text paths for detailed analysis
P_TEXTS_PATH=""
Q_TEXTS_PATH=""
if [ "${DETAILED_ANALYSIS}" = "true" ]; then
  P_TEXTS_PATH="../../data/${DATASET_NAME}/test.csv"

  if [ "${ALGO}" = "condgen_filter" ]; then
    if [ "${DATA_TYPE}" = "intermediate" ]; then
      Q_TEXTS_PATH="../../DPSFT/results/intermediate/generations_${MODE}/generated_${FILE_STEM}.jsonl"
    else
      Q_TEXTS_PATH="../../DPSFT/results/synthetic/generations_${MODE}/generated_${FILE_STEM}.jsonl"
    fi
  elif [ "${ALGO}" = "dpft_filter" ]; then
    if [ "${DATA_TYPE}" = "intermediate" ]; then
      Q_TEXTS_PATH="../../DPSFT/results/intermediate/generations_${MODE}/generated_${FILE_STEM}.jsonl"
    else
      echo "Error: dpft_filter only supports intermediate mode" >&2
      exit 1
    fi
  elif [ "${ALGO}" = "dpft" ] || [ "${ALGO}" = "noft" ]; then
    Q_TEXTS_PATH="../../DPSFT/results/synthetic/generations_${MODE}/generated_${FILE_STEM}.csv"
  else
    Q_TEXTS_PATH="../../DPSFT/results/synthetic/generations_${MODE}/generated_${FILE_STEM}.jsonl"
  fi

  echo "Detailed analysis enabled"
  echo "P texts path: ${P_TEXTS_PATH}"
  echo "Q texts path: ${Q_TEXTS_PATH}"
fi

bash scripts/run_compute_mauve_biorxiv.sh "${EMBED_FILE}" "${SAVE_PATH}" "${DETAILED_ANALYSIS}" "${P_TEXTS_PATH}" "${Q_TEXTS_PATH}"

echo "Job finished at: $(date)"