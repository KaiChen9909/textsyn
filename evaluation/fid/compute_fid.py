"""Compute Fréchet Inception Distance (FID) between two sets of text embeddings.

FID = ||μ_p - μ_q||² + Tr(Σ_p + Σ_q - 2·sqrtm(Σ_p · Σ_q))

Reference: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
Converge to a Local Nash Equilibrium", NeurIPS 2017.
"""

import argparse
import json
import logging
import os
import warnings

import numpy as np
from scipy import linalg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def compute_statistics(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of embeddings."""
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def compute_fid(
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute FID between two Gaussian distributions.

    Args:
        mu_p: Mean of reference distribution, shape (D,).
        sigma_p: Covariance of reference distribution, shape (D, D).
        mu_q: Mean of generated distribution, shape (D,).
        sigma_q: Covariance of generated distribution, shape (D, D).
        eps: Small value added to diagonal for numerical stability.

    Returns:
        FID score (lower is better, 0 means identical distributions).
    """
    diff = mu_p - mu_q
    diff_sq = np.dot(diff, diff)

    # Product of covariances: sigma_p @ sigma_q
    covmean, _ = linalg.sqrtm(sigma_p @ sigma_q, disp=False)

    # Handle numerical issues: sqrtm may return complex matrix
    if np.iscomplexobj(covmean):
        imaginary_norm = np.max(np.abs(covmean.imag))
        if imaginary_norm > 1e-3:
            warnings.warn(
                f"sqrtm produced large imaginary component ({imaginary_norm:.4f}). "
                "This may indicate numerical instability. Results may be unreliable."
            )
        covmean = covmean.real

    # If sqrtm fails (NaN or very negative), add eps to diagonal and retry
    if not np.isfinite(covmean).all():
        logging.warning(
            "sqrtm produced non-finite values. Adding eps to covariance diagonals."
        )
        offset = np.eye(sigma_p.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_p + offset) @ (sigma_q + offset)).real

    trace_term = np.trace(sigma_p) + np.trace(sigma_q) - 2.0 * np.trace(covmean)
    fid = diff_sq + trace_term
    return float(fid)


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID between two sets of text embeddings."
    )
    parser.add_argument(
        "--p_feats_path",
        type=str,
        required=True,
        help="Path to reference embeddings .npy file.",
    )
    parser.add_argument(
        "--q_feats_path",
        type=str,
        required=True,
        help="Path to generated embeddings .npy file.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Directory to save fid.json results. If 'none' or omitted, only print.",
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=None,
        help="Subsample at most this many embeddings from each set (for speed/memory).",
    )
    args = parser.parse_args()

    # Load embeddings
    logging.info(f"Loading reference embeddings from {args.p_feats_path}...")
    p_feats = np.load(args.p_feats_path).astype(np.float64)

    logging.info(f"Loading generated embeddings from {args.q_feats_path}...")
    q_feats = np.load(args.q_feats_path).astype(np.float64)

    logging.info(f"Reference shape: {p_feats.shape}, Generated shape: {q_feats.shape}")

    # Optional subsampling
    if args.max_n is not None:
        if p_feats.shape[0] > args.max_n:
            idx = np.random.choice(p_feats.shape[0], args.max_n, replace=False)
            p_feats = p_feats[idx]
            logging.info(f"Subsampled reference to {p_feats.shape[0]} embeddings.")
        if q_feats.shape[0] > args.max_n:
            idx = np.random.choice(q_feats.shape[0], args.max_n, replace=False)
            q_feats = q_feats[idx]
            logging.info(f"Subsampled generated to {q_feats.shape[0]} embeddings.")

    # Compute statistics
    logging.info("Computing statistics for reference embeddings...")
    mu_p, sigma_p = compute_statistics(p_feats)

    logging.info("Computing statistics for generated embeddings...")
    mu_q, sigma_q = compute_statistics(q_feats)

    # Compute FID
    logging.info("Computing FID...")
    fid_score = compute_fid(mu_p, sigma_p, mu_q, sigma_q)

    logging.info(f"FID: {fid_score:.4f}")

    result = {
        "fid": fid_score,
        "n_reference": int(p_feats.shape[0]),
        "n_generated": int(q_feats.shape[0]),
        "embedding_dim": int(p_feats.shape[1]),
        "p_feats_path": args.p_feats_path,
        "q_feats_path": args.q_feats_path,
    }

    if args.save_path and args.save_path.lower() != "none":
        os.makedirs(args.save_path, exist_ok=True)
        out_file = os.path.join(args.save_path, "fid.json")
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"Results saved to {out_file}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
