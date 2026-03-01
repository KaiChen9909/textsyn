#!/bin/bash

# ==========================================
# SLURM Resource Configuration (for 2 H100 GPUs)
# ==========================================
#SBATCH --job-name=biorxiv_train_filter
#SBATCH --account=NAIRR250463-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=27:00:00
#SBATCH --output=logs/%j_train_filter_stdout.txt

# ==========================================
# Environment Setup
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load anaconda
source activate syn

export HF_HOME="/anvil/scratch/x-kchen28/.cache/huggingface"
cd $SLURM_SUBMIT_DIR

# ==========================================
# Run Training Script
# ==========================================
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# ==========================================
# Argument Parsing
# ==========================================
DATASET_NAME=${1:? "Missing argument: dataset name"}
ALGO=${2:? "Missing argument: algorithm name"}

# Parameters for current training task (can differ from generation settings)
TRAIN_GPU_NUM=${3:-2}
TRAIN_MODEL_NAME=${4:-gemma}
PORT_ID=${5:-29500}

# Parameters for locating generated filtered data (must match gen_filter.sh settings)
PREV_EPS=${6:-4.0}
PREV_USE_DP=${7:-1}
PREV_MODEL_NAME=${8:-gemma}
RHO_FILTER=${9:-0.03}
N_GEN=${10:-5000}
L=${11:-4}

# ==========================================
# Parse Model for Current Training
# ==========================================
if [ "${TRAIN_MODEL_NAME}" = "gemma" ]; then
  GEN_MODEL="google/gemma-3-1b-pt"
  MODEL_STR="gemma-3-1b"
  MODEL_PT="gemma-3-1b-pt"
elif [ "${TRAIN_MODEL_NAME}" = "qwen" ]; then
  GEN_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
  MODEL_STR="qwen2.5-1.5b"
  MODEL_PT="Qwen2.5-1.5B-Instruct"
else
  echo "Error: Unknown train model '${TRAIN_MODEL_NAME}'" >&2
  exit 1
fi

# ==========================================
# Parse Model Used for Data Generation (for constructing file path)
# ==========================================
if [ "${PREV_MODEL_NAME}" = "gemma" ]; then
  PREV_MODEL_PT="gemma-3-1b-pt"
elif [ "${PREV_MODEL_NAME}" = "qwen" ]; then
  PREV_MODEL_PT="Qwen2.5-1.5B-Instruct"
else
  echo "Error: Unknown prev model '${PREV_MODEL_NAME}'" >&2
  exit 1
fi

# ==========================================
# Dataset Hyperparameters (for current training)
# ==========================================
if [ "${DATASET_NAME}" = "biorxiv" ]; then
  BS="1024"
  STEP="320"
  LR="1e-3"
  PORT="${PORT_ID}"
  SEQLEN="512"
  GPUS="${TRAIN_GPU_NUM}"
  MAX_INST_LEN="300"

  # Determine NP based on generation epsilon (for constructing file path)
  if [ "${PREV_EPS}" = "4.0" ]; then
    PREV_NP="4.3"
  else
    echo "Error: NP not defined for prev_eps=${PREV_EPS}. Please add the corresponding NP value." >&2
    exit 1
  fi

  # Determine file name parameters based on whether DP was used during generation
  if [ "${PREV_USE_DP}" = "1" ]; then
    GEN_EPS="${PREV_EPS}"
    GEN_NP="${PREV_NP}"
  else
    GEN_EPS="-1.0"
    GEN_NP="-1"
  fi

else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi


# JSONL Output by gen_filter.sh (using generation model parameters)
FULL_DATASET_NAME="${DATASET_NAME}_${ALGO}"
GEN_FILTER_OUTPUT="results/intermediate/generations_${FULL_DATASET_NAME}/generated_${FULL_DATASET_NAME}_rho-${RHO_FILTER}_model-${PREV_MODEL_PT}_dp-eps-${GEN_EPS}-np-${GEN_NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}.jsonl"


echo "=========================================="
echo "Generated Data Info (from gen_filter.sh):"
echo "  - Prev Model: ${PREV_MODEL_NAME} (${PREV_MODEL_PT})"
echo "  - Prev Eps: ${PREV_EPS}, NP: ${PREV_NP}"
echo "  - Prev Use DP: ${PREV_USE_DP}"
echo "  - RHO Filter: ${RHO_FILTER}"
echo "  - N_GEN: ${N_GEN}"
echo "  - File: ${GEN_FILTER_OUTPUT}"
echo "=========================================="
echo "Current Training Info:"
echo "  - Train Model: ${TRAIN_MODEL_NAME} (${MODEL_PT})"
echo "  - Train GPUs: ${TRAIN_GPU_NUM}"
echo "  - Batch Size: ${BS}, Steps: ${STEP}, LR: ${LR}"
echo "=========================================="

if [ ! -f "${GEN_FILTER_OUTPUT}" ]; then
  echo "Error: Gen filter output not found: ${GEN_FILTER_OUTPUT}" >&2
  echo "Please run gen_filter.sh first to generate filtered training data." >&2
  exit 1
fi

# ==========================================
# Non-DP Training
# ==========================================
DEVICE_BS="8"

bash scripts/train/run_biorxiv_filter-condgen_ft_nondp.sh \
  ${BS} ${STEP} ${LR} ${PORT} ${SEQLEN} ${DEVICE_BS} ${GPUS} \
  ${FULL_DATASET_NAME} "${GEN_FILTER_OUTPUT}" ${GEN_EPS} ${GEN_NP} ${RHO_FILTER} \
  cosine ${GEN_MODEL} ${MODEL_STR} ${N_GEN} ${L} ${MAX_INST_LEN}

echo "Job finished at: $(date)"
