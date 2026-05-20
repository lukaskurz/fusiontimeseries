"""Measure forward pass inference time for trained models on batch 9 test set.

Run with: uv run src/fusiontimeseries/experiments/measure_inference_time.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fusiontimeseries.experiments.config import FinetuningConfig
from fusiontimeseries.experiments.dataset import FluxDataset, prepare_model_input
from fusiontimeseries.experiments.model import get_model
from fusiontimeseries.lib.conditioning import ConditionRegistry
from fusiontimeseries.loralib.layers import BilinearLoRA, Linear, RSSBilinearLoRA


def parse_folder_name(folder_name: str) -> str | None:
    """Parse folder name to extract method.

    Returns the method name if it matches one of the target methods, None otherwise.
    """
    METHODS = [
        "Linear",
        "BilinearLoRA",
        "RSSBilinearLoRA",
        "FullContext-Linear",
        "FullContext-BilinearLoRA",
        "FullContext-RSSBilinearLoRA",
    ]

    for method in METHODS:
        if folder_name.startswith(method):
            return method

    return None


def parse_adapter_type(
    folder_name: str,
) -> type[Linear | BilinearLoRA | RSSBilinearLoRA]:
    """Parse the adapter type from the folder name."""
    if "BilinearLoRA" in folder_name and "RSS" not in folder_name:
        return BilinearLoRA
    elif "RSSBilinearLoRA" in folder_name:
        return RSSBilinearLoRA
    elif "Linear" in folder_name:
        return Linear
    else:
        raise ValueError(
            f"Could not determine adapter type from folder name: {folder_name}"
        )


def load_trained_model(
    output_dir: Path,
    config: FinetuningConfig,
    Adapter: type[Linear | BilinearLoRA | RSSBilinearLoRA],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """Load a trained model from an output directory."""

    # Initialize the model with the same architecture
    model = get_model(
        config=config,
        output_dir=output_dir,  # Not used for loading, but required by signature
        device=device,
        Adapter=Adapter,
    )

    # Load the saved LoRA weights
    weights_path = output_dir / "lora_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    lora_weights = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(lora_weights, strict=False)

    # Ensure the entire model is on the correct device
    model = model.to(device)

    return model.eval()


def measure_forward_pass_time(
    model: nn.Module,
    dataset: FluxDataset,
    config: FinetuningConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    warmup_steps: int = 5,
) -> list[float]:
    """Measure forward pass time for each sample in the dataset.

    Args:
        model: The model to evaluate
        dataset: The test dataset
        config: The finetuning configuration
        device: Device to run inference on
        warmup_steps: Number of warmup forward passes before timing

    Returns:
        List of forward pass times in seconds for each sample
    """
    inference_times = []

    # Get all samples from batch_9
    batch9_data = dataset.flux_data.get("batch_9", {})
    if not batch9_data:
        raise ValueError("batch_9 data not found in dataset")

    # Warmup
    print(f"    Running {warmup_steps} warmup steps...")
    for simulation_id, flux in list(batch9_data.items())[:warmup_steps]:
        sample = prepare_model_input(
            flux=flux,
            cutoff_idx=config.eval_context_cutoff,
            config=config,
        )
        with torch.no_grad():
            with ConditionRegistry.patch(
                op_params=sample["operating_parameters"].unsqueeze(0).to(device)
            ):
                with torch.autocast(device_type=device, dtype=torch.float32):
                    _ = model(
                        input_ts=sample["context"]
                        .unsqueeze(0)
                        .to(device, non_blocking=True),
                        input_padding=sample["context_mask"]
                        .unsqueeze(0)
                        .to(device, non_blocking=True),
                        freq=sample["freq"].unsqueeze(0).to(device, non_blocking=True),
                    )

    # Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure inference time for each sample
    print(f"    Measuring inference time for {len(batch9_data)} samples...")
    for simulation_id, flux in batch9_data.items():
        sample = prepare_model_input(
            flux=flux,
            cutoff_idx=config.eval_context_cutoff,
            config=config,
        )

        # Time the forward pass
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            with ConditionRegistry.patch(
                op_params=sample["operating_parameters"].unsqueeze(0).to(device)
            ):
                with torch.autocast(device_type=device, dtype=torch.float32):
                    predictions = model(
                        input_ts=sample["context"]
                        .unsqueeze(0)
                        .to(device, non_blocking=True),
                        input_padding=sample["context_mask"]
                        .unsqueeze(0)
                        .to(device, non_blocking=True),
                        freq=sample["freq"].unsqueeze(0).to(device, non_blocking=True),
                    )

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        inference_time = end_time - start_time
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
    std_error = (np.std(values_array, ddof=1) / np.sqrt(len(values_array))).item()

    return mean, std_error


def main():
    """Measure and report inference times for all trained models."""
    project_root = Path(__file__).parent.parent.parent.parent
    outputs_dir = project_root / "outputs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Outputs directory: {outputs_dir}")

    # Get all run folders (excluding TEST folders)
    run_folders = [
        f
        for f in outputs_dir.iterdir()
        if f.is_dir() and not f.name.startswith("TEST-")
    ]

    # Group inference times by method
    method_inference_times = defaultdict(list)

    for folder in sorted(run_folders):
        method = parse_folder_name(folder.name)

        # Skip if not one of the target methods
        if method is None:
            continue

        print("\n" + "=" * 80)
        print(f"Processing: {folder.name}")
        print(f"  Method: {method}")

        try:
            # Load the config
            config_path = folder / "fts_config.json"
            if not config_path.exists():
                print("  ⚠️  Config not found, skipping")
                continue

            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = FinetuningConfig(**config_dict)

            # Check if weights exist
            weights_path = folder / "lora_weights.pt"
            if not weights_path.exists():
                print("  ⚠️  Weights not found, skipping")
                continue

            # Parse adapter type from folder name
            Adapter = parse_adapter_type(folder.name)
            print(f"  Adapter type: {Adapter.__name__}")

            # Load the trained model
            print("  Loading model...")
            model = load_trained_model(
                output_dir=folder,
                config=config,
                Adapter=Adapter,
                device=device,
            )

            # Create test dataset with batch_9 namespace
            print("  Loading batch_9 test dataset...")
            test_dataset = FluxDataset(namespaces=["batch_9"], config=config)

            # Measure inference times
            print("  Measuring inference times...")
            inference_times = measure_forward_pass_time(
                model=model,
                dataset=test_dataset,
                config=config,
                device=device,
            )

            # Add to method results
            method_inference_times[method].extend(inference_times)

            mean_time, se_time = calculate_mean_and_error(inference_times)
            print(
                f"  ✓ Mean inference time: {mean_time:.6f} ± {se_time:.6f} seconds ({len(inference_times)} samples)"
            )

            # Clear GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ Error processing {folder.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Calculate and print results
    print("\n" + "=" * 80)
    print("Forward Pass Inference Time on Batch 9 Test Set")
    print("=" * 80)

    METHODS = [
        "Linear",
        "BilinearLoRA",
        "RSSBilinearLoRA",
        "FullContext-Linear",
        "FullContext-BilinearLoRA",
        "FullContext-RSSBilinearLoRA",
    ]

    for method in METHODS:
        if method in method_inference_times:
            mean, error = calculate_mean_and_error(method_inference_times[method])
            print(
                f"{method:30s}: {mean:8.6f} ± {error:8.6f} seconds ({len(method_inference_times[method])} samples)"
            )
        else:
            print(f"{method:30s}: No data")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
