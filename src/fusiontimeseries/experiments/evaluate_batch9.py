# run with uv run src/fusiontimeseries/experiments/evaluate_batch9.py

import json
from pathlib import Path
import torch
from torch import nn

from fusiontimeseries.loralib.layers import BilinearLoRA, Linear, RSSBilinearLoRA
from fusiontimeseries.experiments.config import FinetuningConfig
from fusiontimeseries.experiments.dataset import FluxDataset
from fusiontimeseries.experiments.model import get_model, evaluate


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

    lora_weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(lora_weights, strict=False)

    # Ensure the entire model is on the correct device
    model = model.to(device)

    return model.eval()


def main():
    outputs_dir = Path("./outputs")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all output folders except those starting with "TEST-"
    output_folders = [
        folder
        for folder in outputs_dir.iterdir()
        if folder.is_dir() and not folder.name.startswith("TEST-")
    ]

    print(f"Found {len(output_folders)} output folders to evaluate")

    results_summary = {}

    for output_folder in sorted(output_folders)[10:]:
        folder_name = output_folder.name
        print("\n" + "=" * 80)
        print(f"Evaluating: {folder_name}")
        print("=" * 80)

        try:
            # Load the config
            config_path = output_folder / "fts_config.json"
            if not config_path.exists():
                print(f"  ⚠️  Config not found, skipping: {config_path}")
                continue

            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = FinetuningConfig(**config_dict)

            # Parse adapter type from folder name
            Adapter = parse_adapter_type(folder_name)
            print(f"  Adapter type: {Adapter.__name__}")
            print(
                f"  Config: eval_cutoff={config.eval_context_cutoff}, "
                f"context_len={config.context_length}, subsampling={config.subsampling}"
            )

            # Load the trained model
            print("  Loading model...")
            model = load_trained_model(
                output_dir=output_folder,
                config=config,
                Adapter=Adapter,
                device=device,
            )

            # Create test dataset with batch_9 namespace
            print("  Loading batch_9 test dataset...")
            test_dataset = FluxDataset(namespaces=["batch_9"], config=config)
            print(f"  Test samples: {len(test_dataset)}")

            # Evaluate the model
            print("  Evaluating model...")
            test_results = evaluate(
                model=model,
                config=config,
                data=test_dataset.flux_data,
                device=device,
            )

            # Extract RMSE and SE for batch_9
            batch9_results = test_results.get("batch_9", {})
            rmse = batch9_results.get("rmse", None)
            rmse_se = batch9_results.get("rmse_standard_error", None)

            print(f"  ✓ RMSE: {rmse:.6f} ± {rmse_se:.6f}")

            # Save results to the output folder
            results_file = output_folder / "batch9_test_results.json"
            with open(results_file, "w") as f:
                json.dump(test_results, f, indent=4)
            print(f"  Results saved to: {results_file}")

            # Add to summary
            results_summary[folder_name] = {
                "rmse": rmse,
                "rmse_standard_error": rmse_se,
                "adapter": Adapter.__name__,
                "eval_context_cutoff": config.eval_context_cutoff,
                "context_length": config.context_length,
                "subsampling": config.subsampling,
            }

            # Clear GPU memory
            del model
            torch.cuda.empty_cache()

            # break  # Remove this break to evaluate all folders, currently set to evaluate only the first one for testing

        except Exception as e:
            print(f"  ✗ Error processing {folder_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save summary results
    summary_file = outputs_dir / "batch9_evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=4)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)

    # Print summary table
    print("\nSummary of Results:")
    print(f"{'Folder':<60} {'RMSE':<15} {'±SE':<15}")
    print("-" * 90)
    for folder_name, results in sorted(results_summary.items()):
        rmse = results["rmse"]
        rmse_se = results["rmse_standard_error"]
        print(f"{folder_name:<60} {rmse:<15.6f} {rmse_se:<15.6f}")


if __name__ == "__main__":
    main()
