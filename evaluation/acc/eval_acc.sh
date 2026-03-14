#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH -J biorxiv_eval_acc
#SBATCH -A dplab
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%j_eval_acc_stdout.txt

# ==========================================
# 环境准备
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load uv
source /scratch/pkq2ps/envs/syn/bin/activate

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
SAVE_PATH=${5:-"none"}
GPU_NUM=${6:-2}
MODEL=${7:-gemma}
ACC_MODEL=${8:-"openai-community/gpt2"}  # fixed eval model, independent of generation model
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
  TRAIN_TEXT_COLUMN="generated_text"
  VALIDATION_FILE="../../data/biorxiv/test.csv"
  VALIDATION_TEXT_COLUMN="abstract"
  LR="1e-3"
  SEQLEN="512"
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

  elif [ "${ALGO}" = "noexample"* ]; then
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
      # non-DP: overwrite EPS and NP
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
      # non-DP: overwrite EPS and NP
      EPS="-1.0"
      NP="-1"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
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

  elif [ "${ALGO}" = "dpft" ] || [ "${ALGO}" = "noft" ]; then
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
    echo "Error: Unknown algo '${ALGO}'. Supported: condgen_filter, condgen, dpft_filter, dpft, noexample, noexample_4attr, noexample_8outof24attr, noexample_16attr, noexample_24attr, example50, noft, noft_noexample, condgen_pretrain" >&2
    exit 1
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

# Determine base path for data
BASE_PATH="synthetic"
if [ "${DATA_TYPE}" = "intermediate" ] && { [ "${ALGO}" = "condgen_filter" ] || [ "${ALGO}" = "dpft_filter" ]; }; then
  BASE_PATH="intermediate"
fi

TRAIN_FILE="../../DPSFT/results/${BASE_PATH}/generations_${MODE}/generated_${FILE_STEM}.${FILE_TYPE}"

if [ "${SAVE_PATH}" = "none" ]; then
  OUTPUT_DIR="results/${DATASET_NAME}_acc/${FILE_STEM}"
else
  OUTPUT_DIR="${SAVE_PATH}"
fi

echo "Mode: ${MODE}"
echo "File stem: ${FILE_STEM}"
echo "Base path: ${BASE_PATH}"
echo "Train file: ${TRAIN_FILE}"
echo "Validation file: ${VALIDATION_FILE}"
echo "Output dir: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

python compute_acc.py \
    --model_name_or_path ${ACC_MODEL} \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALIDATION_FILE} \
    --train_text_column ${TRAIN_TEXT_COLUMN} \
    --validation_text_column ${VALIDATION_TEXT_COLUMN} \
    --max_seq_length 512 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --do_train \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-3 \
    --num_train_epochs 5 \
    --save_strategy "no" \
    --eval_strategy "no" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --seed 42 \
    --bf16 \
    --report_to "none"

echo "Job finished at: $(date)"
