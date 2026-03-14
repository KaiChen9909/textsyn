#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH -J biorxiv_eval_self_bleu
#SBATCH -A dplab
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%j_eval_self_bleu_stdout.txt

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

DATASET_NAME=${1:? "Missing argument: dataset name"}
ALGO=${2:?         "Missing argument: algorithm name"}
EPS=${3:-4.0}
USE_DP=${4:-1}
GPU_NUM=${5:-2}
SAVE_PATH=${6:-"none"}

# Self-BLEU 参数
N_GRAM=${7:-4}
SAMPLE_SIZE=${8:-1000}
NUM_WORKERS=${9:-8}

if [ "${DATASET_NAME}" = "biorxiv" ]; then
  MODEL_STR="gemma-3-1b"
  MODEL_PT="gemma-3-1b-pt"
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

  if [ "${ALGO}" = "noexample" ] || [ "${ALGO}" = "noexample_4attr" ] || [ "${ALGO}" = "noexample_8outof24attr" ] || [ "${ALGO}" = "example50" ] || [ "${ALGO}" = "noft_noexample" ]; then
    MAX_INST_LEN="300"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="jsonl"
    TEXT_COLUMN="generated_text"

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
      EPS="1000000000.0"
      NP="0"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    fi

  elif [ "${ALGO}" = "noexample_16attr" ] || [ "${ALGO}" = "noexample_24attr" ]; then
    MAX_INST_LEN="600"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="jsonl"
    TEXT_COLUMN="generated_text"

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
      EPS="1000000000.0"
      NP="0"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"
    fi

  elif [ "${ALGO}" = "dpft" ] || [ "${ALGO}" = "noft" ]; then
    MAX_INST_LEN="32"
    MODE="${DATASET_NAME}_${ALGO}"
    FILE_TYPE="csv"
    TEXT_COLUMN="generated_text"

    if [ "${USE_DP}" = "1" ]; then
      if [ "${EPS}" = "4.0" ]; then
        NP="3.013"
      else
        echo "Error: NP not defined for eps=${EPS}. Please add the corresponding NP value." >&2
        exit 1
      fi
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-nm-${NP}_lr-${LR}_seqlen-${SEQLEN}_n-${N_GEN}"
    else
      # non-DP: overwrite EPS and NP
      EPS="1000000000.0"
      NP="0"
      FILE_STEM="${DATASET_NAME}_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-nm-${NP}_lr-${LR}_seqlen-${SEQLEN}_n-${N_GEN}"
    fi

  else
    echo "Error: Unknown algo '${ALGO}'. Supported: noexample, noexample_4attr, noexample_8outof24attr, noexample_16attr, noexample_24attr, example50, dpft, noft, noft_noexample" >&2
    exit 1
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

# 生成文件路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GEN_DIR="${PROJECT_ROOT}/DPSFT/results/synthetic/generations_${MODE}"
INPUT_FILE="${GEN_DIR}/generated_${FILE_STEM}.${FILE_TYPE}"

# 结果保存路径
if [ "${SAVE_PATH}" = "none" ]; then
  RESULTS_DIR="${SCRIPT_DIR}/results"
else
  RESULTS_DIR="${SAVE_PATH}"
fi
mkdir -p "${RESULTS_DIR}"
OUTPUT_FILE="${RESULTS_DIR}/self_bleu_${FILE_STEM}.json"

echo "Mode: ${MODE}"
echo "File stem: ${FILE_STEM}"
echo "Input file: ${INPUT_FILE}"
echo "Output file: ${OUTPUT_FILE}"
echo "N-gram: ${N_GRAM}"
echo "Sample size: ${SAMPLE_SIZE}"
echo "Num workers: ${NUM_WORKERS}"

# 检查输入文件是否存在
if [ ! -f "${INPUT_FILE}" ]; then
  echo "Error: Input file not found: ${INPUT_FILE}" >&2
  exit 1
fi

# 运行 Self-BLEU 计算
echo "Computing Self-BLEU score..."
python "${SCRIPT_DIR}/compute_self_bleu.py" \
  --input_path "${INPUT_FILE}" \
  --text_column_name "${TEXT_COLUMN}" \
  --n_gram "${N_GRAM}" \
  --sample_size "${SAMPLE_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --save_path "${OUTPUT_FILE}"

echo "Job finished at: $(date)"
