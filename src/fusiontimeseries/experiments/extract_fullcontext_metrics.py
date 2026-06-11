"""Extract training runtime and inference speed metrics for FullContext adapter types."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """Parse folder name to extract method and context length.

    Format: METHOD-CONTEXT_LEN-DATE-RUN_NR
    """
    parts = folder_name.split("-")
    method_parts = []
    context_len = None

    for i, part in enumerate(parts):
        if part.isdigit() and context_len is None:
            context_len = part
            break
        else:
            method_parts.append(part)

    method = "-".join(method_parts)
    return method, context_len  # type: ignore[return-value]


def extract_train_runtime(folder: Path) -> float | None:
    """Extract train_runtime from train_summary.json.

    Args:
        folder: Path to the run folder

    Returns:
        train_runtime value in seconds or None if not found
    """
    train_summary_path = folder / "train_summary.json"
    if not train_summary_path.exists():
        return None

    with open(train_summary_path, "r") as f:
        data = json.load(f)

    return data.get("metrics", {}).get("train_runtime")


def extract_inference_times(folder: Path) -> list[float]:
    """Extract inference times from trainer_state.json.

    Calculates 1 / eval_samples_per_second for each eval step.

    Args:
        folder: Path to the run folder

    Returns:
        List of inference times (1 / eval_samples_per_second)
    """
    trainer_state_path = folder / "checkpoint-3000" / "trainer_state.json"
    if not trainer_state_path.exists():
        return []

    with open(trainer_state_path, "r") as f:
        data = json.load(f)

    inference_times = []
    log_history = data.get("log_history", [])

    for entry in log_history:
        if "eval_samples_per_second" in entry:
            eval_samples_per_second = entry["eval_samples_per_second"]
            if eval_samples_per_second > 0:
                inference_time = 1.0 / eval_samples_per_second
                inference_times.append(inference_time)

    return inference_times


def calculate_mean_and_error(values: list[float]) -> tuple[float, float]:
    """Calculate mean and standard error.

    Args:
        values: List of numerical values

    Returns:
        Tuple of (mean, standard_error)
    """
    if not values:
        return np.nan, np.nan

    values_array = np.array(values)
    mean = np.mean(values_array).item()
    std_error = np.std(values_array, ddof=1) / np.sqrt(len(values_array))

    return mean, std_error


def main():
    """Extract and summarize training metrics by FullContext adapter type."""
    project_root = Path(__file__).parent.parent.parent.parent
    outputs_dir = project_root / "outputs"

    # Get all run folders (excluding TEST folders)
    run_folders = [
        f
        for f in outputs_dir.iterdir()
        if f.is_dir() and not f.name.startswith("TEST-")
    ]

    # Group data by adapter type
    adapter_train_runtimes = defaultdict(list)
    adapter_inference_times = defaultdict(list)

    for folder in run_folders:
        method, context_len = parse_folder_name(folder.name)

        # Skip if not one of the target FullContext adapter types
        if method not in [
            "FullContext-Linear",
            "FullContext-BilinearLoRA",
            "FullContext-RSSBilinearLoRA",
        ]:
            continue

        # Extract train runtime
        train_runtime = extract_train_runtime(folder)
        if train_runtime is not None:
            adapter_train_runtimes[method].append(train_runtime)

        # Extract inference times
        inference_times = extract_inference_times(folder)
        adapter_inference_times[method].extend(inference_times)

    # Calculate and print results
    print("\n" + "=" * 70)
    print("Training Runtime by FullContext Adapter Type")
    print("=" * 70)
    for adapter in [
        "FullContext-Linear",
        "FullContext-BilinearLoRA",
        "FullContext-RSSBilinearLoRA",
    ]:
        if adapter in adapter_train_runtimes:
            mean, error = calculate_mean_and_error(adapter_train_runtimes[adapter])
            print(
                f"{adapter:30s}: {mean:8.2f} ± {error:6.2f} seconds ({len(adapter_train_runtimes[adapter])} runs)"
            )
        else:
            print(f"{adapter:30s}: No data")

    print("\n" + "=" * 70)
    print("Inference Time (1 / eval_samples_per_second) by FullContext Adapter Type")
    print("=" * 70)
    for adapter in [
        "FullContext-Linear",
        "FullContext-BilinearLoRA",
        "FullContext-RSSBilinearLoRA",
    ]:
        if adapter in adapter_inference_times:
            mean, error = calculate_mean_and_error(adapter_inference_times[adapter])
            print(
                f"{adapter:30s}: {mean:8.6f} ± {error:8.6f} seconds ({len(adapter_inference_times[adapter])} samples)"
            )
        else:
            print(f"{adapter:30s}: No data")

    print("\n" + "=" * 70)

    # Return data for further processing if needed
    return {
        "train_runtime": {
            adapter: calculate_mean_and_error(runtimes)
            for adapter, runtimes in adapter_train_runtimes.items()
        },
        "inference_time": {
            adapter: calculate_mean_and_error(times)
            for adapter, times in adapter_inference_times.items()
        },
    }


if __name__ == "__main__":
    results = main()
