import argparse
import logging
import sys
from typing import Tuple

# --- Third-Party Library Imports ---
from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant

# --- Setup Basic Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def find_sigma_for_dpsgd(
    total_epsilon: float,
    total_delta: float,
    subsampling_rate: float,
    iterations: int,
    sigma_search_min: float = 1e-5,
    sigma_search_max: float = 1000.0,
    search_precision: int = 50,
) -> float:
  """Finds the required sigma for DP-SGD.

  Args:
      total_epsilon: The target epsilon.
      total_delta: The target delta.
      subsampling_rate: The sampling rate (q).
      iterations: The number of composition steps.
      sigma_search_min: The lower bound for the binary search on sigma.
      sigma_search_max: The upper bound for the binary search on sigma.
      search_precision: The number of iterations for the binary search.

  Returns:
      The calculated sigma for the DP-SGD mechanism.
  """
  # Objective function for the search
  def get_total_epsilon(sigma: float,  subsampling_rate: float, iterations: int) -> float:
    """Calculates the total epsilon for a given sigma, sampling rate, iter."""
    accountant = rdp_privacy_accountant.RdpAccountant()

    subsampled_event = dp_event.PoissonSampledDpEvent(
        sampling_probability=subsampling_rate,
        event=dp_event.GaussianDpEvent(noise_multiplier=sigma),
    )
    accountant.compose(subsampled_event, count=iterations)
    return accountant.get_epsilon(target_delta=total_delta)

  # Binary search for sigma
  low = sigma_search_min
  high = sigma_search_max

  for _ in range(search_precision):
    mid_sigma = (low + high) / 2
    try:
      epsilon_guess = get_total_epsilon(mid_sigma, subsampling_rate, iterations)
    except (OverflowError, ValueError):
      # If accountant fails (e.g., due to very low sigma), treat as infinite epsilon
      epsilon_guess = float("inf")

    if epsilon_guess > total_epsilon:
      low = mid_sigma
    else:
      high = mid_sigma

  # 'high' is the lowest sigma that meets the budget
  return high


def main():
  """Main function to parse arguments and run the calculation."""
  parser = argparse.ArgumentParser(
      description="Calculate sigma for DP-SGD."
  )
  parser.add_argument(
      "--total_epsilon",
      type=float,
      required=True,
      help="Target epsilon for the combined process.",
  )
  parser.add_argument(
      "--total_delta",
      type=float,
      required=True,
      help="Target delta for the combined process.",
  )
  parser.add_argument(
      "--dataset_size",
      type=int,
      required=True,
      help="Dataset size.",
  )
  parser.add_argument(
      "--batch_size", type=int, required=True, help="Batch size for DP-SGD."
  )
  parser.add_argument(
      "--iterations",
      type=int,
      required=True,
      help="Number of iterations for DP-SGD.",
  )
  args = parser.parse_args()

  sampling_rate = args.batch_size / args.dataset_size

  logging.info("Target Budget:")
  logging.info(f"  - Epsilon: {args.total_epsilon}")
  logging.info(f"  - Delta:   {args.total_delta}\n")
  logging.info("Known Subsampled Gaussian Parameters:")
  logging.info(
      "  - Sampling Rate (q):"
      f" {sampling_rate:.4f} ({args.batch_size}/{args.dataset_size})"
  )
  logging.info(f"  - Iterations:        {args.iterations}")

  # --- Find Sigma for DP-SGD ---
  required_sigma = find_sigma_for_dpsgd(
      total_epsilon=args.total_epsilon,
      total_delta=args.total_delta,
      subsampling_rate=sampling_rate,
      iterations=args.iterations
  )

  # --- Verification and Final Output ---
  logging.info("=" * 40)
  logging.info("          RESULTS & VERIFICATION")
  logging.info("=" * 40)
  logging.info(f"Required Sigma: {required_sigma:.6f}")

  # Verify the calculation
  final_accountant = rdp_privacy_accountant.RdpAccountant()
  subsampled_event2 = dp_event.PoissonSampledDpEvent(
      sampling_probability=sampling_rate,
      event=dp_event.GaussianDpEvent(noise_multiplier=required_sigma),
  )
  final_accountant.compose(subsampled_event2, count=args.iterations)
  final_epsilon = final_accountant.get_epsilon(target_delta=args.total_delta)

  logging.info(
      f"Verification: Total Epsilon after composition: {final_epsilon:.4f}"
  )
  logging.info("=" * 40)


if __name__ == "__main__":
  main()
