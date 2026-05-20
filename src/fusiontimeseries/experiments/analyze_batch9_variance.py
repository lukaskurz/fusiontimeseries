"""Analyze the variance of batch 9 test set tail means.

This script calculates the variance and standard error of the mean energy flux
values from the last 240 timesteps of each simulation in the batch 9 test set.
It also computes the average within-trajectory standard error.

Run with: uv run src/fusiontimeseries/experiments/analyze_batch9_variance.py
"""

import json
from pathlib import Path

import numpy as np


def main():
    """Calculate variance and standard error of batch 9 test set tail means."""

    # Load flux data
    flux_data_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "flux_data.json"
    )

    print(f"Loading flux data from: {flux_data_path}")
    with open(flux_data_path, "r") as f:
        all_flux_data = json.load(f)

    # Get batch_9 data (NOT subsampled - use the full trajectories)
    batch9_data = all_flux_data.get("batch_9", {})

    if not batch9_data:
        raise ValueError("batch_9 data not found in flux_data.json")

    print(f"Found {len(batch9_data)} simulations in batch_9")

    # Calculate the mean and standard error of residuals for last 240 timesteps for each simulation
    tail_length = 240
    tail_means = []
    tail_standard_errors = []
    tail_mean_absolute_residuals = []
    tail_rms_residuals = []

    print(
        f"\nCalculating tail statistics (last {tail_length} timesteps) for each simulation:"
    )
    print("=" * 80)

    for sim_id, flux in sorted(batch9_data.items()):
        energy_flux = np.array(flux["energy_flux"])

        # Get the last 240 timesteps
        tail = energy_flux[-tail_length:]

        # Calculate mean
        tail_mean = np.mean(tail)
        tail_means.append(tail_mean)

        # Calculate residuals (actual - mean)
        residuals = tail - tail_mean

        # Calculate standard error of residuals
        std_dev_residuals = np.std(residuals, ddof=1)
        std_error_residuals = std_dev_residuals / np.sqrt(len(tail))
        tail_standard_errors.append(std_error_residuals)

        # Calculate mean absolute residual
        mean_abs_residual = np.mean(np.abs(residuals))
        tail_mean_absolute_residuals.append(mean_abs_residual)

        # Calculate RMS residual
        rms_residual = np.sqrt(np.mean(residuals**2))
        tail_rms_residuals.append(rms_residual)

        print(
            f"Simulation {sim_id:>6s}: {len(energy_flux):4d} timesteps, tail mean = {tail_mean:8.2f}, MAE = {mean_abs_residual:6.4f}, RMS = {rms_residual:6.4f}"
        )

    # Convert to numpy arrays for statistical calculations
    tail_means_array = np.array(tail_means)
    tail_standard_errors_array = np.array(tail_standard_errors)
    tail_mean_absolute_residuals_array = np.array(tail_mean_absolute_residuals)
    tail_rms_residuals_array = np.array(tail_rms_residuals)

    # Calculate statistics for tail means
    overall_mean = np.mean(tail_means_array)
    variance = np.var(tail_means_array, ddof=1)  # Sample variance (unbiased)
    std_dev = np.std(tail_means_array, ddof=1)  # Sample standard deviation
    std_error_of_means = std_dev / np.sqrt(
        len(tail_means_array)
    )  # Standard error of the mean

    min_mean = np.min(tail_means_array)
    max_mean = np.max(tail_means_array)
    median_mean = np.median(tail_means_array)

    # Calculate average standard error of residuals across all trajectories
    avg_std_error_residuals = np.mean(tail_standard_errors_array)
    min_std_error = np.min(tail_standard_errors_array)
    max_std_error = np.max(tail_standard_errors_array)
    median_std_error = np.median(tail_standard_errors_array)

    # Calculate average residual metrics across all trajectories
    avg_mae_residuals = np.mean(tail_mean_absolute_residuals_array)
    min_mae = np.min(tail_mean_absolute_residuals_array)
    max_mae = np.max(tail_mean_absolute_residuals_array)
    median_mae = np.median(tail_mean_absolute_residuals_array)

    avg_rms_residuals = np.mean(tail_rms_residuals_array)
    min_rms = np.min(tail_rms_residuals_array)
    max_rms = np.max(tail_rms_residuals_array)
    median_rms = np.median(tail_rms_residuals_array)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Batch 9 Test Set: Tail Statistics (Last 240 Timesteps)")
    print("=" * 80)
    print(f"Number of simulations:        {len(tail_means_array)}")
    print(f"Tail length:                  {tail_length} timesteps")
    print("\n--- Tail Mean Statistics ---")
    print(f"Mean of tail means:           {overall_mean:.6f}")
    print(f"Variance of tail means:       {variance:.6f}")
    print(f"Standard deviation:           {std_dev:.6f}")
    print(f"Standard error of means:      {std_error_of_means:.6f}")
    print(f"\nMinimum tail mean:            {min_mean:.6f}")
    print(f"Maximum tail mean:            {max_mean:.6f}")
    print(f"Median tail mean:             {median_mean:.6f}")
    print(f"Range:                        {max_mean - min_mean:.6f}")
    print(f"Coefficient of variation:     {(std_dev / overall_mean) * 100:.2f}%")
    print("\n--- Average Within-Trajectory Residual Metrics ---")
    print(f"Average MAE (Mean Abs Error): {avg_mae_residuals:.6f}")
    print(f"  Min MAE:                    {min_mae:.6f}")
    print(f"  Max MAE:                    {max_mae:.6f}")
    print(f"  Median MAE:                 {median_mae:.6f}")
    print(f"\nAverage RMS residual:         {avg_rms_residuals:.6f}")
    print(f"  Min RMS:                    {min_rms:.6f}")
    print(f"  Max RMS:                    {max_rms:.6f}")
    print(f"  Median RMS:                 {median_rms:.6f}")
    print(f"\nAverage SE of residuals:      {avg_std_error_residuals:.6f}")
    print(f"  Min SE:                     {min_std_error:.6f}")
    print(f"  Max SE:                     {max_std_error:.6f}")
    print(f"  Median SE:                  {median_std_error:.6f}")
    print("=" * 80)

    # Additional quartile information
    q25 = np.percentile(tail_means_array, 25)
    q75 = np.percentile(tail_means_array, 75)
    iqr = q75 - q25

    print("\nQuartile Analysis (Tail Means):")
    print(f"25th percentile (Q1):         {q25:.6f}")
    print(f"50th percentile (Median):     {median_mean:.6f}")
    print(f"75th percentile (Q3):         {q75:.6f}")
    print(f"Interquartile range (IQR):    {iqr:.6f}")
    print("=" * 80)

    # Distribution summary
    print("\nDistribution Summary (Tail Means):")
    print(f"Mean ± 1 SD:  [{overall_mean - std_dev:.2f}, {overall_mean + std_dev:.2f}]")
    print(
        f"Mean ± 2 SD:  [{overall_mean - 2 * std_dev:.2f}, {overall_mean + 2 * std_dev:.2f}]"
    )
    print(
        f"Mean ± 1 SE:  [{overall_mean - std_error_of_means:.2f}, {overall_mean + std_error_of_means:.2f}]"
    )
    print("=" * 80)

    # Save results to JSON
    output_path = (
        Path(__file__).parent.parent.parent.parent
        / "outputs"
        / "batch9_tail_variance_analysis.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "tail_length": tail_length,
        "n_simulations": len(tail_means_array),
        "tail_means": tail_means_array.tolist(),
        "tail_standard_errors": tail_standard_errors_array.tolist(),
        "tail_mean_absolute_residuals": tail_mean_absolute_residuals_array.tolist(),
        "tail_rms_residuals": tail_rms_residuals_array.tolist(),
        "statistics": {
            "tail_means": {
                "mean": float(overall_mean),
                "variance": float(variance),
                "std_dev": float(std_dev),
                "std_error_of_means": float(std_error_of_means),
                "min": float(min_mean),
                "max": float(max_mean),
                "median": float(median_mean),
                "range": float(max_mean - min_mean),
                "coefficient_of_variation_percent": float(
                    (std_dev / overall_mean) * 100
                ),
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(iqr),
            },
            "within_trajectory_mean_absolute_errors": {
                "average": float(avg_mae_residuals),
                "min": float(min_mae),
                "max": float(max_mae),
                "median": float(median_mae),
            },
            "within_trajectory_rms_residuals": {
                "average": float(avg_rms_residuals),
                "min": float(min_rms),
                "max": float(max_rms),
                "median": float(median_rms),
            },
            "within_trajectory_standard_errors": {
                "average": float(avg_std_error_residuals),
                "min": float(min_std_error),
                "max": float(max_std_error),
                "median": float(median_std_error),
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
