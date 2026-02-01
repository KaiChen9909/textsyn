set -x
prompt_len=${6:-512}
seq_len=${7:-512}
prompt_str=${8:-biorxiv-complex8et-conditions}
prompt_file=${9:-../AIM/results/synthetic_biorxiv_example50_et_5k_rho-0.18_iter-2000.csv}
n_gen=${10:-5000}
model_str=${11:-gemma-3-1b-pt}
python generation_biorxiv_condgen.py \
    -m results/outputs/$1/model_epoch$2 \
    -pl ${prompt_len} -sl ${seq_len} -d 0 -o ${prompt_str} -ps ${prompt_str} \
    -out generated_${prompt_str}_model-${model_str}_dp-eps-$3-np-$4-lr-$5_seqlen-${prompt_len}-${seq_len}_temp-1.0_tp-0.95_tk-0_eval_n-${n_gen}.jsonl \
    -n_gen ${n_gen} -bs 512 -tp 0.95 -pf ${prompt_file}
