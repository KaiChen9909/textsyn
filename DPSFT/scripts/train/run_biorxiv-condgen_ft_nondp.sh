set -x
seq_len=${5:-512}
device_bs=${6:-4}
gpus=${7:-2}
dataset_name=${8:-biorxiv-condgen}
max_inst_len=${9:-300}

if [[ "${dataset_name}" == "biorxiv-complex8et-condgen" ]]; then
  prompt_style="biorxiv-condgen"
else
  prompt_style="${dataset_name}"
fi

lr_scheduler=${10:-constant}
model_name=${11:-google/gemma-3-1b-pt}
model_str=${12:-gemma-3-1b}
python generate_train_command.py \
  --dataset_name ${dataset_name} \
  --model_name ${model_name} \
  --job_sess ${model_str}_${dataset_name}_nondp_bs-$1_step-$2_lr-$3-${lr_scheduler}_seed-42 \
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

