# Author: Krishna Pillutla
# License: GPLv3
# 
# Modifications:
#   - Simplified the code logic to compute MAUVE on loaded embeddings.

import argparse
import json
import logging
import math
import os
import time
from types import SimpleNamespace
import faiss
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import auc as compute_area_under_curve
from sklearn.preprocessing import normalize
import time

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def compute_mauve(
    p_features=None,
    q_features=None,
    num_buckets='auto',
    pca_max_data=-1,
    kmeans_explained_var=0.9,
    kmeans_num_redo=5,
    kmeans_max_iter=500,
    divergence_curve_discretization_size=25,
    mauve_scaling_factor=5,
    verbose=False,
    seed=25,
    return_labels=False,
):
  """Compute the MAUVE score between two text generations P and Q.

  P is specified as ``p_features``. Same with Q.

  :param ``p_features``: ``numpy.ndarray`` of shape (n, d), where n is the
  number of generations.
  :param ``q_features``: ``numpy.ndarray`` of shape (n, d), where n is the
  number of generations.
  :param ``num_buckets``: the size of the histogram to quantize P and Q.
  Options: ``'auto'`` (default, which is n/10) or an integer.
  :param ``pca_max_data``: the number data points to use for PCA. If `-1`, use
  all the data. Default -1.
  :param ``kmeans_explained_var``: amount of variance of the data to keep in
  dimensionality reduction by PCA. Default 0.9.
  :param ``kmeans_num_redo``: number of times to redo k-means clustering (the
  best objective is kept). Default 5.
      Try reducing this to 1 in order to reduce running time.
  :param ``kmeans_max_iter``: maximum number of k-means iterations. Default 500.
      Try reducing this to 100 in order to reduce running time.
  :param ``divergence_curve_discretization_size``: Number of points to consider
  on the divergence curve. Default 25.
      Larger values do not offer much of a difference.
  :param ``mauve_scaling_factor``: The constant``c`` from the paper. Default 5.
      See `Best Practices <index.html#best-practices-for-mauve>`_ for details.
  :param ``verbose``: If True, print running time updates.
  :param ``seed``: random seed to initialize k-means cluster assignments.
  :param ``return_labels``: If True, also return cluster labels for each sample.

  :return: an object with fields p_hist, q_hist, divergence_curve and mauve.

  * ``out.mauve`` is a number between 0 and 1, the MAUVE score. Higher values
  means P is closer to Q.
  * ``out.frontier_integral``, a number between 0 and 1. Lower values mean that
  P is closer to Q.
  * ``out.p_hist`` is the obtained histogram for P. Same for ``out.q_hist``.
  * ``out.divergence_curve`` contains the points in the divergence curve. It is
  of shape (m, 2), where m is ``divergence_curve_discretization_size``
  """

  if p_features is None:
    raise ValueError('Supply p_features')
  if q_features is None:
    raise ValueError('Supply q_features')
  p_features = get_features_from_input(
      p_features,
  )
  q_features = get_features_from_input(
      q_features,
  )
  if num_buckets == 'auto':
    # heuristic: use num_clusters = num_generations / 10
    num_buckets = max(
        2, int(round(min(p_features.shape[0], q_features.shape[0]) / 10))
    )
  elif not isinstance(num_buckets, int):
    raise ValueError('num_buckets is expected to be an integer or "auto"')

  # Acutal binning
  t1 = time.time()
  cluster_result = cluster_feats(
      p_features,
      q_features,
      num_clusters=num_buckets,
      norm='l2',
      whiten=False,
      pca_max_data=pca_max_data,
      explained_variance=kmeans_explained_var,
      num_redo=kmeans_num_redo,
      max_iter=kmeans_max_iter,
      seed=seed,
      verbose=verbose,
      return_labels=return_labels,
  )
  if return_labels:
    p, q, p_labels, q_labels = cluster_result
  else:
    p, q = cluster_result
  t2 = time.time()
  if verbose:
    logging.info(f'total discretization time: {round(t2 - t1, 2)} seconds')

  # Divergence curve and mauve
  mixture_weights = np.linspace(
      1e-6, 1 - 1e-6, divergence_curve_discretization_size
  )
  divergence_curve = get_divergence_curve_for_multinomials(
      p, q, mixture_weights, mauve_scaling_factor
  )
  x, y = divergence_curve.T
  idxs1 = np.argsort(x)
  idxs2 = np.argsort(y)
  mauve_score = 0.5 * (
      compute_area_under_curve(x[idxs1], y[idxs1])
      + compute_area_under_curve(y[idxs2], x[idxs2])
  )
  fi_score = get_fronter_integral(p, q)
  to_return = SimpleNamespace(
      p_hist=p,
      q_hist=q,
      divergence_curve=divergence_curve,
      mauve=mauve_score,
      frontier_integral=fi_score,
      num_buckets=num_buckets,
  )
  if return_labels:
    to_return.p_labels = p_labels
    to_return.q_labels = q_labels
  return to_return


def get_features_from_input(
    features,
):
  features = np.asarray(features)
  return features


def cluster_feats(
    p,
    q,
    num_clusters,
    norm='none',
    whiten=True,
    pca_max_data=-1,
    explained_variance=0.9,
    num_redo=5,
    max_iter=500,
    seed=0,
    verbose=False,
    return_labels=False,
):
  assert 0 < explained_variance < 1
  if verbose:
    logging.info(f'seed = {seed}')
  assert norm in ['none', 'l2', 'l1', None]
  data1 = np.vstack([q, p])
  if norm in ['l2', 'l1']:
    data1 = normalize(data1, norm=norm, axis=1)
  pca = PCA(n_components=None, whiten=whiten, random_state=seed + 1)
  if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
    pca.fit(data1)
  elif 0 < pca_max_data < data1.shape[0]:
    rng = np.random.RandomState(seed + 5)
    idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
    pca.fit(data1[idxs])
  else:
    raise ValueError(
        f'Invalid argument pca_max_data={pca_max_data} with'
        f' {data1.shape[0]} datapoints'
    )
  s = np.cumsum(pca.explained_variance_ratio_)
  idx = np.argmax(s >= explained_variance)  # last index to consider
  if verbose:
    logging.info(f'performing clustering in lower dimension = {idx}')
  data1 = pca.transform(data1)[:, : idx + 1]
  # Cluster features and obtain the labels for each data point.
  data1 = data1.astype(np.float32)  # Faiss requires float32.
  t1 = time.time()
  kmeans = faiss.Kmeans(
      data1.shape[1],
      num_clusters,
      niter=max_iter,
      verbose=verbose,
      nredo=num_redo,
      update_index=True,
      seed=seed + 2,
  )
  kmeans.train(data1)
  _, labels = kmeans.index.search(data1, 1)
  labels = labels.reshape(-1)
  t2 = time.time()
  if verbose:
    logging.info(f'kmeans time: {round(t2 - t1, 2)} s')

  q_labels = labels[: len(q)]
  p_labels = labels[len(q) :]

  q_bins = np.histogram(
      q_labels, bins=num_clusters, range=[0, num_clusters], density=True
  )[0]
  p_bins = np.histogram(
      p_labels, bins=num_clusters, range=[0, num_clusters], density=True
  )[0]

  if return_labels:
    return p_bins / p_bins.sum(), q_bins / q_bins.sum(), p_labels, q_labels
  else:
    return p_bins / p_bins.sum(), q_bins / q_bins.sum()


def kl_multinomial(p, q):
  assert p.shape == q.shape
  if np.logical_and(p != 0, q == 0).any():
    return np.inf
  else:
    idxs = np.logical_and(p != 0, q != 0)
    return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))


def get_divergence_curve_for_multinomials(
    p, q, mixture_weights, scaling_factor
):
  divergence_curve = [[0, np.inf]]  # extreme point
  for w in np.sort(mixture_weights):
    r = w * p + (1 - w) * q
    divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
  divergence_curve.append([np.inf, 0])  # other extreme point
  return np.exp(-scaling_factor * np.asarray(divergence_curve))


def get_fronter_integral(p, q, scaling_factor=2):
  total = 0.0
  for p1, q1 in zip(p, q):
    if p1 == 0 and q1 == 0:
      pass
    elif p1 == 0:
      total += q1 / 4
    elif q1 == 0:
      total += p1 / 4
    elif abs(p1 - q1) > 1e-8:
      t1 = p1 + q1
      t2 = p1 * q1 * (math.log(p1) - math.log(q1)) / (p1 - q1)
      total += 0.25 * t1 - 0.5 * t2
    # else: contribution is 0
  return total * scaling_factor


def load_texts(file_path):
  """Load texts from a file. Supports .txt, .jsonl, and .csv formats.

  Special handling: If file_path ends with 'test.csv' and contains 'biorxiv',
  automatically loads both validation.csv and test.csv and merges them.
  """
  # Special case: biorxiv test.csv should include valid.csv
  if file_path.endswith('test.csv') and 'biorxiv' in file_path:
    validation_path = file_path.replace('test.csv', 'valid.csv')
    if os.path.exists(validation_path):
      logging.info(f'Auto-loading validation set from: {validation_path}')
      validation_texts = load_texts_single_file(validation_path)
      logging.info(f'Auto-loading test set from: {file_path}')
      test_texts = load_texts_single_file(file_path)
      texts = validation_texts + test_texts
      logging.info(f'Combined {len(validation_texts)} validation + {len(test_texts)} test = {len(texts)} total')
      return texts

  # Default: load single file
  return load_texts_single_file(file_path)


def load_texts_single_file(file_path):
  """Load texts from a single file."""
  texts = []
  ext = os.path.splitext(file_path)[1].lower()

  if ext == '.txt':
    with open(file_path, 'r', encoding='utf-8') as f:
      texts = [line.strip() for line in f if line.strip()]
  elif ext == '.jsonl':
    with open(file_path, 'r', encoding='utf-8') as f:
      for line in f:
        if line.strip():
          data = json.loads(line)
          # Assume the text is in a field called 'text', 'content', or 'generated_text'
          text = data.get('text') or data.get('content') or data.get('generated_text') or data.get('abstract', '')
          texts.append(text)
  elif ext == '.csv':
    import csv
    with open(file_path, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      for row in reader:
        # Try common field names for text content
        text = row.get('text') or row.get('content') or row.get('generated_text') or row.get('abstract', '')
        texts.append(text)
  else:
    raise ValueError(f'Unsupported file format: {ext}. Supported: .txt, .jsonl, .csv')

  return texts


def compute_shannon_entropy(hist):
  """Compute Shannon entropy of a probability distribution."""
  # Filter out zero probabilities
  hist_nonzero = hist[hist > 0]
  return -np.sum(hist_nonzero * np.log(hist_nonzero))


def compute_detailed_analysis(result, p_texts, q_texts):
  """Compute detailed MAUVE attribution analysis.

  Args:
    result: SimpleNamespace object returned by compute_mauve (with labels)
    p_texts: list of reference texts (human)
    q_texts: list of generated texts (machine)

  Returns:
    dict: Detailed analysis results including macro metrics and extreme clusters
  """
  p_hist = result.p_hist
  q_hist = result.q_hist
  p_labels = result.p_labels
  q_labels = result.q_labels

  # 1. Compute Total Variation Distance (TVD)
  tvd = 0.5 * np.sum(np.abs(p_hist - q_hist))

  # 2. Compute Shannon Entropy
  p_entropy = compute_shannon_entropy(p_hist)
  q_entropy = compute_shannon_entropy(q_hist)

  # 3. Compute cluster delta (q - p)
  cluster_delta = q_hist - p_hist

  # 4. Find extreme clusters
  # Top-5 "machine-overrepresented" clusters (highest positive delta)
  top_machine_indices = np.argsort(cluster_delta)[-5:][::-1]
  # Top-5 "human-exclusive" clusters (most negative delta)
  top_human_indices = np.argsort(cluster_delta)[:5]

  # 5. Extract texts for extreme clusters
  machine_clusters = []
  for idx in top_machine_indices:
    cluster_id = int(idx)
    delta = float(cluster_delta[idx])
    p_count = int(np.sum(p_labels == cluster_id))
    q_count = int(np.sum(q_labels == cluster_id))

    # Get sample texts from this cluster
    q_cluster_texts = [q_texts[i] for i in range(len(q_texts)) if q_labels[i] == cluster_id]
    p_cluster_texts = [p_texts[i] for i in range(len(p_texts)) if p_labels[i] == cluster_id]

    machine_clusters.append({
      'cluster_id': cluster_id,
      'delta': delta,
      'p_prob': float(p_hist[idx]),
      'q_prob': float(q_hist[idx]),
      'p_count': p_count,
      'q_count': q_count,
      'sample_q_texts': q_cluster_texts[:10],  # First 10 samples
      'sample_p_texts': p_cluster_texts[:10],  # First 10 samples
      'total_q_texts': len(q_cluster_texts),
      'total_p_texts': len(p_cluster_texts),
    })

  human_clusters = []
  for idx in top_human_indices:
    cluster_id = int(idx)
    delta = float(cluster_delta[idx])
    p_count = int(np.sum(p_labels == cluster_id))
    q_count = int(np.sum(q_labels == cluster_id))

    # Get sample texts from this cluster
    q_cluster_texts = [q_texts[i] for i in range(len(q_texts)) if q_labels[i] == cluster_id]
    p_cluster_texts = [p_texts[i] for i in range(len(p_texts)) if p_labels[i] == cluster_id]

    human_clusters.append({
      'cluster_id': cluster_id,
      'delta': delta,
      'p_prob': float(p_hist[idx]),
      'q_prob': float(q_hist[idx]),
      'p_count': p_count,
      'q_count': q_count,
      'sample_q_texts': q_cluster_texts[:10],  # First 10 samples
      'sample_p_texts': p_cluster_texts[:10],  # First 10 samples
      'total_q_texts': len(q_cluster_texts),
      'total_p_texts': len(p_cluster_texts),
    })

  # 6. Assemble the analysis report
  analysis = {
    'macro_metrics': {
      'tvd': float(tvd),
      'p_entropy': float(p_entropy),
      'q_entropy': float(q_entropy),
      'entropy_diff': float(q_entropy - p_entropy),
    },
    'extreme_clusters': {
      'top_machine_overrepresented': machine_clusters,
      'top_human_exclusive': human_clusters,
    },
    'metadata': {
      'num_clusters': int(result.num_buckets),
      'num_p_samples': len(p_texts),
      'num_q_samples': len(q_texts),
    },
  }

  return analysis

def main():
  # --- Parse Arguments ---
  parser = argparse.ArgumentParser(
      description='Compute MAUVE score between two sets of features or texts.'
  )
  parser.add_argument(
      '--p_feats_path',
      type=str,
      required=True,
      help='Path to the numpy file containing p features.',
  )
  parser.add_argument(
      '--q_feats_path',
      type=str,
      required=True,
      help='Path to the numpy file containing q features.',
  )
  parser.add_argument(
      '--num_buckets',
      type=int,
      default=-1,
      help='Number of buckets for histogram. Default is "auto".',
  )
  parser.add_argument(
      '--save_path',
      type=str,
      default=None,
      help='path of results saving',
  )
  parser.add_argument(
      '--detailed_mauve_analysis',
      action='store_true',
      default=False,
      help='Whether to perform fine-grained MAUVE attribution analysis and export results.',
  )
  parser.add_argument(
      '--p_texts_path',
      type=str,
      default=None,
      help='Path to the file containing p texts (required for detailed analysis).',
  )
  parser.add_argument(
      '--q_texts_path',
      type=str,
      default=None,
      help='Path to the file containing q texts (required for detailed analysis).',
  )
  args = parser.parse_args()

  # --- Validate detailed analysis arguments ---
  if args.detailed_mauve_analysis:
    if args.p_texts_path is None or args.q_texts_path is None:
      raise ValueError(
          '--detailed_mauve_analysis requires --p_texts_path and --q_texts_path to be specified.'
      )
    if not os.path.exists(args.p_texts_path):
      raise FileNotFoundError(f'P texts file not found: {args.p_texts_path}')
    if not os.path.exists(args.q_texts_path):
      raise FileNotFoundError(f'Q texts file not found: {args.q_texts_path}')

  # --- Load Features ---
  p_feats = np.load(args.p_feats_path)
  q_feats = np.load(args.q_feats_path)
  logging.info(f'p_feats shape: {p_feats.shape}')
  logging.info(f'q_feats shape: {q_feats.shape}')

  # --- Load Texts (if needed for detailed analysis) ---
  p_texts = None
  q_texts = None
  if args.detailed_mauve_analysis:
    p_texts = load_texts(args.p_texts_path)
    q_texts = load_texts(args.q_texts_path)
    logging.info(f'Loaded {len(p_texts)} p texts and {len(q_texts)} q texts')
    if len(p_texts) != p_feats.shape[0]:
      raise ValueError(
          f'Number of p texts ({len(p_texts)}) does not match number of p features ({p_feats.shape[0]})'
      )
    if len(q_texts) != q_feats.shape[0]:
      raise ValueError(
          f'Number of q texts ({len(q_texts)}) does not match number of q features ({q_feats.shape[0]})'
      )

  # --- Compute MAUVE ---
  t = time.time()
  result = compute_mauve(
      p_feats,
      q_feats,
      num_buckets='auto' if args.num_buckets == -1 else args.num_buckets,
      return_labels=args.detailed_mauve_analysis,
  )

  # --- Log Results ---
  logging.info(f'MAUVE: {result.mauve}')
  logging.info(f'Time taken: {time.time() - t}')

  if args.save_path == "none": args.save_path = None
  if args.save_path:
    os.makedirs(args.save_path, exist_ok=True)
    file_path = os.path.join(args.save_path, 'mauve.json')
    results_dict = {
        'mauve': float(result.mauve),
        'frontier_integral': float(result.frontier_integral),
        'num_buckets': int(result.num_buckets),
        'p_feats_path': args.p_feats_path,
        'q_feats_path': args.q_feats_path,
        'p_hist': result.p_hist.tolist(),
        'q_hist': result.q_hist.tolist(),
        'divergence_curve': result.divergence_curve.tolist(),
    }

    with open(file_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logging.info(f'Results saved to: {file_path}')

    # Save detailed analysis if requested
    if args.detailed_mauve_analysis:
      logging.info('Computing detailed MAUVE attribution analysis...')
      detailed_analysis = compute_detailed_analysis(result, p_texts, q_texts)
      detailed_file_path = os.path.join(args.save_path, 'mauve_detailed_analysis.json')

      with open(detailed_file_path, 'w') as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)
      logging.info(f'Detailed analysis saved to: {detailed_file_path}')

      # Log summary statistics
      logging.info('=' * 60)
      logging.info('DETAILED ANALYSIS SUMMARY')
      logging.info('=' * 60)
      logging.info(f"TVD (Total Variation Distance): {detailed_analysis['macro_metrics']['tvd']:.4f}")
      logging.info(f"P Entropy (Human): {detailed_analysis['macro_metrics']['p_entropy']:.4f}")
      logging.info(f"Q Entropy (Machine): {detailed_analysis['macro_metrics']['q_entropy']:.4f}")
      logging.info(f"Entropy Diff (Q - P): {detailed_analysis['macro_metrics']['entropy_diff']:.4f}")
      logging.info('')
      logging.info('Top-5 Machine-Overrepresented Clusters:')
      for i, cluster in enumerate(detailed_analysis['extreme_clusters']['top_machine_overrepresented'], 1):
        logging.info(f"  {i}. Cluster {cluster['cluster_id']}: "
                    f"delta={cluster['delta']:.4f}, "
                    f"p_prob={cluster['p_prob']:.4f}, "
                    f"q_prob={cluster['q_prob']:.4f}, "
                    f"samples(p/q)={cluster['p_count']}/{cluster['q_count']}")
      logging.info('')
      logging.info('Top-5 Human-Exclusive Clusters:')
      for i, cluster in enumerate(detailed_analysis['extreme_clusters']['top_human_exclusive'], 1):
        logging.info(f"  {i}. Cluster {cluster['cluster_id']}: "
                    f"delta={cluster['delta']:.4f}, "
                    f"p_prob={cluster['p_prob']:.4f}, "
                    f"q_prob={cluster['q_prob']:.4f}, "
                    f"samples(p/q)={cluster['p_count']}/{cluster['q_count']}")
      logging.info('=' * 60)

if __name__ == '__main__':
  main()
