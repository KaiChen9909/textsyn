set -x
mode=${3:-biorxiv-condgen}
ext=${4:-csv}
prefix=${5:-''}
text_column=${6:-generated_text}
base_path=${7:-synthetic}  # New parameter: "synthetic" or "intermediate"
python embed_biorxiv.py \
    --input_path ../../DPSFT/results/${base_path}/generations_${mode}/${prefix}generated_$1.${ext} \
    --output_embedding_path embeddings/${prefix}$1_specter2_len-512_embeddings_rerun.npy \
    --text_column_name ${text_column} \
    --batch_size 128 \
    --device $2