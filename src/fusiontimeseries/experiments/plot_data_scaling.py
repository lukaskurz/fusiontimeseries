"""
Plot data scaling experiments: test set RMSE vs. number of training batches.

This script analyzes the DataScaling experiments and creates a log-log plot showing
how test set performance improves with additional training data.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_batch_count(folder_name: str) -> int:
    """
    Extract the number of batches from folder name.

    Examples:
        DataScaling-b1-... -> 1 batch
        DataScaling-b12-... -> 2 batches
        DataScaling-b123-... -> 3 batches
        DataScaling-b123457810-... -> 8 batches

    Args:
        folder_name: Name of the DataScaling experiment folder

    Returns:
        Number of batches used in training
    """
    # Extract the batch string between "b" and the next "-"
    parts = folder_name.split("-")
    if len(parts) < 2 or not parts[1].startswith("b"):
        raise ValueError(f"Invalid folder name format: {folder_name}")

    batch_str = parts[1][1:]  # Remove the 'b' prefix

    # Count the number of digits (each digit represents one batch)
    # b1 = 1 batch, b12 = 2 batches, b123 = 3 batches, etc.
    batch_count = len(batch_str)

    return batch_count


def load_test_rmse(folder_path: Path) -> tuple[float, float]:
    """
    Load test set RMSE and standard error from batch9_test_results.json.

    Args:
        folder_path: Path to the experiment folder

    Returns:
        Tuple of (rmse, standard_error)
    """
    results_file = folder_path / "batch9_test_results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        results = json.load(f)

    rmse = results["batch_9"]["rmse"]
    rmse_se = results["batch_9"]["rmse_standard_error"]

    return rmse, rmse_se


def main():
    """Main function to generate data scaling plot."""

    # Find all DataScaling experiment folders
    outputs_dir = Path(__file__).parent.parent.parent.parent / "outputs"
    data_scaling_folders = sorted(outputs_dir.glob("DataScaling-b*-RSSBilinearLoRA-*"))

    print(f"Found {len(data_scaling_folders)} DataScaling experiments\n")

    # Extract batch counts and RMSE values
    batch_counts = []
    rmse_values = []
    rmse_errors = []

    for folder in data_scaling_folders:
        try:
            batch_count = parse_batch_count(folder.name)
            rmse, rmse_se = load_test_rmse(folder)

            batch_counts.append(batch_count)
            rmse_values.append(rmse)
            rmse_errors.append(rmse_se)

            print(f"{folder.name}")
            print(f"  Batches: {batch_count}, RMSE: {rmse:.4f} ± {rmse_se:.4f}")

        except Exception as e:
            print(f"Error processing {folder.name}: {e}")
            continue

    if not batch_counts:
        print("No valid data found!")
        return

    # Sort by batch count
    sorted_indices = np.argsort(batch_counts)
    batch_counts = np.array(batch_counts)[sorted_indices]
    rmse_values = np.array(rmse_values)[sorted_indices]
    rmse_errors = np.array(rmse_errors)[sorted_indices]

    print("\nCreating plot...")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot with error bars
    ax.errorbar(
        batch_counts,
        rmse_values,
        yerr=rmse_errors,
        fmt="o-",
        markersize=8,
        linewidth=2,
        capsize=5,
        capthick=2,
        color="steelblue",
        ecolor="lightsteelblue",
    )

    # Set linear x-axis, log y-axis
    ax.set_yscale("log")
    # ax.set_xscale("log")

    # Set x-axis ticks to show each batch count
    ax.set_xticks(batch_counts)
    ax.set_xticklabels([str(int(bc)) for bc in batch_counts])

    # Set y-axis to show regular numbers (not scientific notation)
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(lambda y, _: f"{y:.1f}")
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)

    # Labels and title
    ax.set_xlabel("Number of Training Batches", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test Set RMSE (log)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Data Scaling (RSSBilinearLoRA): Test Performance vs. Training Data Size",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Format tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Add text annotation for improvement
    # if len(batch_counts) >= 2:
    #     improvement = (1 - rmse_values[-1] / rmse_values[0]) * 100
    #     ax.text(
    #         0.05,
    #         0.05,
    #         f"RMSE reduction: {improvement:.1f}%\n"
    #         f"({batch_counts[0]} → {batch_counts[-1]} batches)",
    #         transform=ax.transAxes,
    #         fontsize=11,
    #         verticalalignment="bottom",
    #         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    #     )

    plt.tight_layout()

    # Create output folder
    output_folder = outputs_dir / "data_scaling_analysis"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save the figure
    output_path = output_folder / "data_scaling_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF
    output_path_pdf = output_folder / "data_scaling_plot.pdf"
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"PDF saved to: {output_path_pdf}")

    # Save summary statistics to JSON
    summary = {
        "experiment_type": "DataScaling-RSSBilinearLoRA",
        "num_experiments": len(batch_counts),
        "batch_counts": batch_counts.tolist(),
        "rmse_values": rmse_values.tolist(),
        "rmse_standard_errors": rmse_errors.tolist(),
        "improvement_percent": float((1 - rmse_values[-1] / rmse_values[0]) * 100),
        "scaling_factor": float(rmse_values[0] / rmse_values[-1]),
    }

    summary_path = output_folder / "data_scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    plt.show()

    return summary


if __name__ == "__main__":
    results = main()
    if results:
        print("\n=== Summary ===")
        print(f"RMSE improvement: {results['improvement_percent']:.1f}%")
        print(f"Scaling factor: {results['scaling_factor']:.2f}x")
