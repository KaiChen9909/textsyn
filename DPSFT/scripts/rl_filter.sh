#!/bin/bash

# ==========================================
# SLURM Resource Configuration
# ==========================================
#SBATCH --job-name=biorxiv_rl_filter
#SBATCH --account=CIS260108-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=27:00:00
#SBATCH --output=logs/%j_rl_filter_stdout.txt

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
# Run KTO Training Script
# ==========================================
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# ==========================================
# Argument Parsing
# ==========================================
DATASET_NAME=${1:? "Missing argument: dataset name"}
ALGO=${2:? "Missing argument: algorithm name"}

# Parameters for current training task
TRAIN_GPU_NUM=${3:-4}
TRAIN_MODEL_NAME=${4:-gemma}
PORT_ID=${5:-29500}

# Parameters for locating generated full data (must match gen_filter.sh settings)
PREV_EPS=${6:-4.0}
PREV_USE_DP=${7:-1}
PREV_MODEL_NAME=${8:-gemma}
RHO_FILTER=${9:-0.03}
N_GEN=${10:-5000}
L=${11:-4}
EPOCH_ID=${12:-79}

# KTO-specific hyperparameters
KTO_LR=${13:-5e-6}
KTO_EPOCHS=${14:-1}
KTO_BETA=${15:-0.1}
KTO_BS=${16:-4}
KTO_GRAD_ACCUM=${17:-4}

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
# Dataset Hyperparameters
# ==========================================
if [ "${DATASET_NAME}" = "biorxiv" ]; then
  # These are for constructing the gen_filter output path (matching gen_filter.sh)
  BS="2048"
  STEP="1120"
  LR="1e-3"
  SEQLEN="512"
  GPUS_GEN="4"  # GPUs used during condgen training
  MAX_INST_LEN="300"
  EPOCH="3"
  LR_VAL="0.001"
  DELTA="3.38e-06"
  CLIP="1.0"

  # Determine NP based on generation epsilon
  if [ "${PREV_EPS}" = "4.0" ]; then
    PREV_NP="4.3"
  elif [ "${PREV_EPS}" = "1.0" ]; then
    PREV_NP="13.8"
  else
    echo "Error: NP not defined for prev_eps=${PREV_EPS}." >&2
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

  # Prompt style for KTO
  PROMPT_STYLE="${DATASET_NAME}_condgen_filter_generation"

else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

# ==========================================
# Construct paths
# ==========================================
FULL_DATASET_NAME="${DATASET_NAME}_${ALGO}"

# Full labeled JSONL from gen_filter.sh (with "full_" prefix)
FULL_DATASET="results/intermediate/generations_${FULL_DATASET_NAME}/full_generated_${FULL_DATASET_NAME}_rho-${RHO_FILTER}_model-${PREV_MODEL_PT}_dp-eps-${GEN_EPS}-np-${GEN_NP}-lr-${LR}_seqlen-${MAX_INST_LEN}-${SEQLEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}-L-${L}.jsonl"

# Reconstruct the condgen model directory (the model used in gen_filter.sh)
LR_SCHEDULER_CONDGEN="constant"
if [ "${PREV_USE_DP}" = "1" ]; then
  CONDGEN_JOB_SESS="${MODEL_STR}_${DATASET_NAME}_condgen_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER_CONDGEN}_seed-42"
  CONDGEN_DIR="${CONDGEN_JOB_SESS}_biorxiv_condgen_noredacted_model${PREV_MODEL_PT}_eps${GEN_EPS}_delta${DELTA}_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip${CLIP}_np${GEN_NP}_gpus${GPUS_GEN}"
else
  CONDGEN_JOB_SESS="${MODEL_STR}_${DATASET_NAME}_condgen_nondp_bs-${BS}_step-${STEP}_lr-${LR}-${LR_SCHEDULER_CONDGEN}_seed-42"
  CONDGEN_DIR="${CONDGEN_JOB_SESS}_biorxiv_condgen_noredacted_model${PREV_MODEL_PT}_eps-1.0_delta0.1_bs${BS}_maxseq${MAX_INST_LEN}-${SEQLEN}_epoch${EPOCH}_lr${LR_VAL}_clip-1.0_np-1_gpus${GPUS_GEN}"
fi

MODEL_PATH="results/outputs/${CONDGEN_DIR}/model_epoch${EPOCH_ID}"

# Output directory for KTO model
KTO_SESS="${MODEL_STR}_${DATASET_NAME}_${ALGO}_kto_preveps${GEN_EPS}-np${GEN_NP}-rho${RHO_FILTER}-n${N_GEN}-L${L}_lr-${KTO_LR}_beta-${KTO_BETA}_epoch-${KTO_EPOCHS}_seed-42"
OUTPUT_DIR="results/outputs/${KTO_SESS}"

# ==========================================
# Display configuration
# ==========================================
echo "=========================================="
echo "Full Labeled Data (from gen_filter.sh):"
echo "  - File: ${FULL_DATASET}"
echo "=========================================="
echo "Source Model (condgen model from gen_filter.sh):"
echo "  - Model: ${TRAIN_MODEL_NAME} (${MODEL_PT})"
echo "  - Path: ${MODEL_PATH}"
echo "=========================================="
echo "KTO Training Config:"
echo "  - Learning Rate: ${KTO_LR}"
echo "  - Epochs: ${KTO_EPOCHS}"
echo "  - Beta: ${KTO_BETA}"
echo "  - Per-device BS: ${KTO_BS}"
echo "  - Grad Accum: ${KTO_GRAD_ACCUM}"
echo "  - GPUs: ${TRAIN_GPU_NUM}"
echo "  - Output: ${OUTPUT_DIR}"
echo "=========================================="

# ==========================================
# Validate inputs
# ==========================================
if [ ! -f "${FULL_DATASET}" ]; then
  echo "Error: Full labeled dataset not found: ${FULL_DATASET}" >&2
  echo "Please run gen_filter.sh first to generate the full labeled dataset." >&2
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Error: Model path not found: ${MODEL_PATH}" >&2
  echo "Please ensure the condgen model exists (run merge_model.sh if needed)." >&2
  exit 1
fi

# ==========================================
# Determine accelerate config
# ==========================================
if [ "${TRAIN_GPU_NUM}" = "1" ]; then
  ACCEL_CFG="accelerate_configs/accelerate_config_nofsdp.cfg"
else
  ACCEL_CFG="accelerate_configs/accelerate_config_nofsdp_gpu${TRAIN_GPU_NUM}.cfg"
fi

# ==========================================
# Run KTO Training
# ==========================================
set -x

accelerate launch \
    --main_process_port ${PORT_ID} \
    --config_file ${ACCEL_CFG} \
    train_kto.py \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_path ${FULL_DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_style ${PROMPT_STYLE} \
    --max_prompt_length ${MAX_INST_LEN} \
    --max_completion_length ${SEQLEN} \
    --learning_rate ${KTO_LR} \
    --num_train_epochs ${KTO_EPOCHS} \
    --per_device_train_batch_size ${KTO_BS} \
    --gradient_accumulation_steps ${KTO_GRAD_ACCUM} \
    --beta ${KTO_BETA} \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --seed 42 \
    2>&1 | tee -a results/logs/${KTO_SESS}.txt

echo "Job finished at: $(date)"
