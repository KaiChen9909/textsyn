set -x
delta=${7}
seq_len=${8:-512}
device_bs=${9:-4}
gpus=${10:-2}
dataset_name=${11:-biorxiv}
lr_scheduler=${12:-constant}
model_name=${13:-google/gemma-3-1b-pt}
model_str=${14:-gemma-3-1b}
max_inst_len=32

python generate_train_command.py \
  --dataset_name ${dataset_name} \
  --model_name ${model_name} \
  --job_sess ${model_str}_${dataset_name}_dpft_bs-$1_step-$2_lr-$3-${lr_scheduler}_seed-42 \
  --eps $5 \
  --noise_multiplier $6 \
  --delta ${delta} \
  --clip 1 \
  --perdevice_bs ${device_bs} \
  --gpus ${gpus} \
  --max_instruction_len ${max_inst_len} \
  --max_answer_len ${seq_len} \
  --total_bs $1 \
  --num_steps $2 \
  --lr $3 \
  --lr_scheduler ${lr_scheduler} \
  --prompt_style ${dataset_name}_generation \
  --main_process_port $4 \
  --seed 42

