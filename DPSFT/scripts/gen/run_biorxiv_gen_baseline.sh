set -x
if [ "$1" = "noft" ]; then
  MODEL_PATH="google/gemma-3-1b-pt"
else
  MODEL_PATH="results/outputs/$1/model_epoch$2"
fi

rho_filter=${7:-""}
seq_len=${8:-512}
output_str=${9:-biorxiv}
prompt_str='biorxiv' # no prompt so fixed as biorxiv
n_gen=${10:-5000}
model_str=${11:-gemma-3-1b-pt}
prev_n_gen=${15:-""}
L=${16:-""}

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

python generation_biorxiv_gen.py \
    -m ${MODEL_PATH} \
    -l 512 -d 0 -o ${output_str} -ps ${prompt_str} \
    -out generated_${output_str}_model-${model_str}_dp-eps-$3-nm-$4_lr-$5-step-$6${rho_str}${prev_n_gen_str}${L_str}_seqlen-512_n-${n_gen}.csv \
    -n_gen ${n_gen} -bs 512 -tp 0.95