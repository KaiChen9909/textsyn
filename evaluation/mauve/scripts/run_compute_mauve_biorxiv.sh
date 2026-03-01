set -x

# Arguments:
# $1: embedding file name (e.g., biorxiv_noexample_model-gemma-3-1b-pt_dp-eps-4.0-np-4.3-lr-1e-3_seqlen-300-512_temp-1.0_tp-0.95_tk-0_eval_n-5000_specter2_len-512_embeddings_rerun.npy)
# $2: save path (e.g., results/ or "none")
# $3: (optional) detailed_mauve_analysis flag (true/false, default: false)
# $4: (optional) p_texts_path (required if $3 is true)
# $5: (optional) q_texts_path (required if $3 is true)

EMBED_FILE=$1
SAVE_PATH=$2
DETAILED_ANALYSIS=${3:-false}
P_TEXTS_PATH=${4:-""}
Q_TEXTS_PATH=${5:-""}

CMD="python compute_mauve.py \
    --p_feats_path embeddings/biorxiv_valid_test__specter2_len-512_embeddings.npy \
    --q_feats_path embeddings/${EMBED_FILE} \
    --save_path ${SAVE_PATH}"

# Add detailed analysis arguments if enabled
if [ "${DETAILED_ANALYSIS}" = "true" ]; then
    if [ -z "${P_TEXTS_PATH}" ] || [ -z "${Q_TEXTS_PATH}" ]; then
        echo "Error: --detailed_mauve_analysis requires P_TEXTS_PATH and Q_TEXTS_PATH to be specified."
        exit 1
    fi
    CMD="${CMD} --detailed_mauve_analysis --p_texts_path ${P_TEXTS_PATH} --q_texts_path ${Q_TEXTS_PATH}"
fi

eval ${CMD}