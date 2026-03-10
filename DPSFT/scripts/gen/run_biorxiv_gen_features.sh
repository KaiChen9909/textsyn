set -x

rho_filter=${7:-""}
prompt_len=${8:-512}
seq_len=${9:-512}
output_str=${10:-biorxiv-complex8et-conditions}
prompt_str=${11:-biorxiv-complex8et-conditions}
prompt_file=${12:-../AIM/results/synthetic_biorxiv_example50_et_5k_rho-0.18_iter-2000.csv}
n_gen=${13:-5000}
model_str=${14:-gemma-3-1b-pt}
prev_n_gen=${15:-""}
L=${16:-""}
schema_column=${17:-"generated_text"}

# Build optional rho filter string
rho_str=""
if [ -n "${rho_filter}" ]; then
  rho_str="_rho-${rho_filter}"
fi

if [ -n "${prev_n_gen}" ]; then
  prev_n_gen_str="_n${prev_n_gen}"
fi

if [ -n "${L}" ]; then
  L_str="_L${L}"
fi

if [ "$1" = "noft" ]; then
  if [ "${model_str}" = "gemma-3-1b-pt" ]; then
    MODEL_PATH="google/gemma-3-1b-pt"
    BS=512
  elif [ "${model_str}" = "gemma-3-27b-pt" ]; then
    MODEL_PATH="google/gemma-3-27b-pt"
    BS=48
  fi
else
  BS=512
  MODEL_PATH="results/outputs/$1/model_epoch$2"
fi

python generation_biorxiv_condgen.py \
    -m ${MODEL_PATH} \
    -pl ${prompt_len} -sl ${seq_len} -d 0 -o ${output_str} -ps ${prompt_str} \
    -out generated_${output_str}_model-${model_str}_dp-eps-$3-np-$4-lr-$5-step-$6${rho_str}${prev_n_gen_str}${L_str}_seqlen-${prompt_len}-${seq_len}_temp-1.0_tp-0.95_tk-0_eval_n-${n_gen}.jsonl \
    -n_gen ${n_gen} -bs ${BS} -tp 0.95 -pf ${prompt_file} -sc ${schema_column}