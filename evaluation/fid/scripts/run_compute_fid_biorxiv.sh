set -x
python compute_fid.py \
    --p_feats_path ../mauve/embeddings/biorxiv_valid_test__specter2_len-512_embeddings.npy \
    --q_feats_path $1 \
    --save_path $2
