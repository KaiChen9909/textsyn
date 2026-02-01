set -x
python compute_mauve.py \
    --p_feats_path embeddings/biorxiv_valid_test__specter2_len-512_embeddings.npy \
    --q_feats_path embeddings/$1