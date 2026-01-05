"""This file holds the benchmarking dataset and dataloader."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy import stats
import torch

__all__ = [
    "IN_DISTRIBUTION_ITERATIONS",
    "OUT_OF_DISTRIBUTION_ITERATIONS",
    "BenchmarkDataProvider",
    "rmse_with_standard_error",
    "Utils",
    "BenchmarkConfig",
]


class BenchmarkConfig(BaseModel):
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_slug: str
    model_prediction_length: int
    relevant_prediction_tail: int
    start_context_length: int


IN_DISTRIBUTION_ITERATIONS: list[
    Literal[
        "iteration_8_ifft",
        "iteration_115_ifft",
        "iteration_131_ifft",
        "iteration_148_ifft",
        "iteration_235_ifft",
        "iteration_262_ifft",
    ]
] = [
    "iteration_8_ifft",
    "iteration_115_ifft",
    "iteration_131_ifft",
    "iteration_148_ifft",
    "iteration_235_ifft",
    "iteration_262_ifft",
]

OUT_OF_DISTRIBUTION_ITERATIONS: list[
    Literal[
        "ood_iteration_0_ifft_realpotens",
        "ood_iteration_1_ifft_realpotens",
        "ood_iteration_2_ifft_realpotens",
        "ood_iteration_3_ifft_realpotens",
        "ood_iteration_4_ifft_realpotens",
    ]
] = [
    "ood_iteration_0_ifft_realpotens",
    "ood_iteration_1_ifft_realpotens",
    "ood_iteration_2_ifft_realpotens",
    "ood_iteration_3_ifft_realpotens",
    "ood_iteration_4_ifft_realpotens",
]


class Utils:
    @staticmethod
    def median_forecast(forecast: torch.Tensor) -> torch.Tensor:
        """Compute the median forecast from the forecast tensor.

        Args:
            forecast (torch.Tensor): Forecast tensor of shape [N, prediction_length, n_quantiles].

        Returns:
            torch.Tensor: Median forecast tensor of shape [N, prediction_length].
        """
        n_quantiles = forecast.shape[-1]
        median_index = n_quantiles // 2
        return forecast[:, :, median_index]


class BenchmarkDataProvider:
    """DataLoader for autoregressive forecasting tasks.

    The first context window is 80 timesteps.
    After forecasting 64 timesteps, the next context window is 80 + 64 = 144 timesteps.
    """

    FLUX_DATA_PATH = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "data"
        / "flux"
        / "benchmark"
        / "flux_data.json"
    )

    def __init__(self) -> None:
        self.flux_data: dict = json.load(open(self.FLUX_DATA_PATH, "r"))

    def get_id(
        self,
        iteration: Literal[
            "iteration_8_ifft",
            "iteration_115_ifft",
            "iteration_131_ifft",
            "iteration_148_ifft",
            "iteration_235_ifft",
            "iteration_262_ifft",
        ],
    ) -> torch.Tensor:
        """Get flux data for a specific iteration.

        Args:
            iteration (str): The iteration to get data for.
        """
        return torch.tensor(
            self.flux_data["in_distribution"][iteration], dtype=torch.float32
        )

    def get_ood(
        self,
        iteration: Literal[
            "ood_iteration_0_ifft_realpotens",
            "ood_iteration_1_ifft_realpotens",
            "ood_iteration_2_ifft_realpotens",
            "ood_iteration_3_ifft_realpotens",
            "ood_iteration_4_ifft_realpotens",
        ],
    ) -> torch.Tensor:
        """Get out-of-distribution flux data for a specific iteration.

        Args:
            iteration (str): The iteration to get data for.
        """
        return torch.tensor(
            self.flux_data["out_of_distribution"][iteration], dtype=torch.float32
        )


def rmse_with_standard_error(y_true, y_pred) -> tuple[float, float]:
    """
    Compute RMSE with standard error using Delta method approximation.

    Parameters
    ----------
    y_true : array-like
        Ground truth values (averaged last 80 targets)
    y_pred : array-like
        Predicted values (averaged last 80 predictions)

    Returns
    -------
    rmse : float
        Root Mean Squared Error
    se_rmse : float
        Standard error of RMSE via Delta method
    """
    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    # Standard error of the mean squared error
    se_mse = stats.sem(squared_errors)

    # Delta method approximation for SE of RMSE
    # SE(√MSE) ≈ SE(MSE) / (2√MSE)
    se_rmse = se_mse / (2 * rmse)

    return rmse, se_rmse


if __name__ == "__main__":
    ##############################
    # Example usage
    ##############################

    provider = BenchmarkDataProvider()
    trace = provider.get_id("iteration_262_ifft")
    print("Trace shape:", trace.shape)
