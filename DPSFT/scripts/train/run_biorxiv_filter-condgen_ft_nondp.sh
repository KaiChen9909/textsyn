set -x
seq_len=${5:-512}
device_bs=${6:-4}
gpus=${7:-2}
dataset_name=${8:-biorxiv_condgen_filter}
gen_filter_path=${9}
gen_eps=${10:-4.0}
gen_np=${11:-4.3}
rho_filter=${12:-0.03}
lr_scheduler=${13:-cosine}
model_name_or_path=${14:-google/gemma-3-1b-pt}
model_str=${15:-gemma-3-1b}
n_gen=${16:-5000}
L=${17:-4}
max_inst_len=${18:-300}

# For condgen_filter with schema, we need to use a different prompt style
# Use the base prompt style without _generation suffix (will be added by generate_train_command.py)
prompt_style="${dataset_name}"

# Extract model type from model_name_or_path for directory naming
# If it's a path (contains 'results/' or 'model'), extract the model type
if [[ "${model_name_or_path}" == *"results/"* ]] || [[ "${model_name_or_path}" == *"/model"* ]]; then
  # It's a local path - use model_str to construct model type name
  if [ "${model_str}" = "gemma-3-1b" ]; then
    model_name="gemma-3-1b-pt"
  elif [ "${model_str}" = "qwen2.5-1.5b" ]; then
    model_name="Qwen2.5-1.5B-Instruct"
  else
    model_name="${model_str}"
  fi
  # Pass both model_name (for naming) and model_path (for loading)
  MODEL_PATH_ARG="--model_path ${model_name_or_path}"
else
  # It's a HuggingFace model name - use as-is
  model_name="${model_name_or_path}"
  MODEL_PATH_ARG=""
fi

python generate_train_command.py \
  --dataset_name ${dataset_name} \
  --dataset_path ${gen_filter_path} \
  --dataset_size ${n_gen} \
  --model_name ${model_name} \
  ${MODEL_PATH_ARG} \
  --job_sess ${model_str}_${dataset_name}_preveps${gen_eps}-np${gen_np}-rho${rho_filter}-n${n_gen}-L${L}_nondp_bs-$1_step-$2_lr-$3-${lr_scheduler}_seed-42 \
  --eps "-1" \
  --delta 0.1 \
  --clip "-1" \
  --perdevice_bs ${device_bs} \
  --gpus ${gpus} \
  --max_instruction_len ${max_inst_len} \
  --max_answer_len ${seq_len} \
  --total_bs $1 \
  --num_steps $2 \
  --lr $3 \
  --lr_scheduler ${lr_scheduler} \
  --prompt_style ${prompt_style}_generation \
  --main_process_port $4 \
  --seed 42
