set -x
seq_len=${6:512}
n_gen=${7:-5000}
model_str=${8:-gemma-3-1b-pt}
prompt_str='biorxiv'
python generation_biorxiv_gen.py \
    -m results/outputs/$1/model_epoch$2 \
    -l 512 -d 0 -o ${prompt_str} -ps ${prompt_str} \
    -out generated_biorxiv_model-${model_str}_dp-eps-$3-nm-$4_lr-$5_seqlen-512_n-${n_gen}.csv \
    -n_gen ${n_gen} -bs 512 -tp 0.95