from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from fusiontimeseries.experiments.config import FinetuningConfig
from fusiontimeseries.experiments.dataset import Namespace
from typing import Any


def plot_forecast(
    results: dict[Namespace, dict[str, Any]],
    simulations: list[str],
    config: FinetuningConfig,
    output_dir: Path,
    show_plots: bool = True,
):
    for namespace, res in results.items():
        for simulation_id in simulations:
            if (
                simulation_id not in res["ground_truths"]
                or simulation_id not in res["forecasts"]
            ):
                print(
                    f"Warning: Simulation ID {simulation_id} not found in results for {namespace}, skipping plot."
                )
                continue

            plt.figure(figsize=(8, 3))
            plt.plot(res["ground_truths"][simulation_id], label="Ground Truth")
            plt.plot(
                range(config.eval_context_cutoff, len(res["forecasts"][simulation_id])),
                res["forecasts"][simulation_id][config.eval_context_cutoff :],
                label="Forecast",
            )
            plt.axvline(
                x=config.eval_context_cutoff,
                color="gray",
                linestyle="--",
                label="Context Cutoff",
            )

            prediction_tail_length: int = 80 if config.subsampling else 240
            gt_mean = np.array(
                res["ground_truths"][simulation_id][-prediction_tail_length:]
            ).mean()
            fc_mean = np.array(
                res["forecasts"][simulation_id][-prediction_tail_length:]
            ).mean()
            plt.title(
                f"{namespace} ({simulation_id}) Forecast ({fc_mean:.2f}) vs Ground Truth ({gt_mean:.2f})"
            )
            plt.xlabel("Time Steps")
            plt.ylabel("Flux")
            plt.grid(axis="both", linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{namespace}_{simulation_id}_forecast.png")
            if show_plots:
                plt.show()
