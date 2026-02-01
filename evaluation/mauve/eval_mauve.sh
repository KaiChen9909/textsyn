#!/bin/bash

# ==========================================
# SLURM 资源配置
# ==========================================
#SBATCH --job-name=biorxiv_gen
#SBATCH --account=NAIRR250463-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_gen_stdout.txt
#SBATCH --error=logs/%j_gen_stderr.txt

# ==========================================
# 环境准备
# ==========================================
mkdir -p logs

# load anaconda and activate env
module load anaconda
source activate syn

cd $SLURM_SUBMIT_DIR

# ==========================================
# 运行你的 Bash 脚本任务
# ==========================================
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# ==========================================
# 参数解析: sbatch gen.sh <dataset_name> <algo> [eps]
#   dataset_name: biorxiv
#   algo:         noexample / example50 / dpft
#   eps:          epsilon value (default: 4.0)
# ==========================================
DATASET_NAME=${1:? "Usage: sbatch gen.sh <dataset_name> <algo> [eps]"}
ALGO=${2:?         "Missing argument: algo (noexample, example50, dpft)"}
EPS=${3:-4.0}

if [ "${DATASET_NAME}" = "biorxiv" ]; then
  MODEL_STR="gemma-3-1b"
  MODEL_PT="gemma-3-1b-pt"
  DELTA="3.38e-06"
  BS="2048"
  STEP="1120"
  LR="1e-3"
  EPOCH="3"
  LR_VAL="0.001"
  GPUS="2"
  EPOCH_ID="79"

  N_GEN=5000
  PROMPT_LEN=300
  SEQ_LEN=512

  if [ "${ALGO}" = "noexample" ]; then
    NP="4.3"
    MODE="biorxiv_${ALGO}"
    FILE_TYPE="jsonl"
    FILE_STEM="biorxiv_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${PROMPT_LEN}-${SEQ_LEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"

  elif [ "${ALGO}" = "example50" ]; then
    NP="4.3"
    MODE="biorxiv_${ALGO}"
    FILE_TYPE="jsonl"
    FILE_STEM="biorxiv_${ALGO}_model-${MODEL_PT}_dp-eps-${EPS}-np-${NP}-lr-${LR}_seqlen-${PROMPT_LEN}-${SEQ_LEN}_temp-1.0_tp-0.95_tk-0_eval_n-${N_GEN}"

  elif [ "${ALGO}" = "dpft" ]; then
    NP="3.013"
    MODE="biorxiv"
    FILE_TYPE="csv"
    FILE_STEM="biorxiv_model-${MODEL_PT}_dp-eps-${EPS}-nm-${NP}_lr-${LR}_seqlen-${SEQ_LEN}_n-${N_GEN}"
    
  else
    echo "Error: Unknown algo '${ALGO}'. Supported: noexample, example50, dpft" >&2
    exit 1
  fi
else
  echo "Error: Unknown dataset '${DATASET_NAME}'. Supported: biorxiv" >&2
  exit 1
fi

GPU_DEVICE=0
EMBED_FILE="${FILE_STEM}_specter2_len-512_embeddings_rerun.npy"

echo "Embedding file: generated_${FILE_STEM}.${FILE_TYPE} in generations_${MODE}/"
bash scripts/run_embed_biorxiv.sh "${FILE_STEM}" "${GPU_DEVICE}" "${MODE}" "${FILE_TYPE}"

echo "Computing MAUVE score..."
bash scripts/run_compute_mauve_biorxiv.sh "${EMBED_FILE}"

echo "Job finished at: $(date)"