from datetime import datetime
from functools import cache
import json
from pathlib import Path
from typing import Literal
from sklearn.model_selection import train_test_split

import numpy as np
from numpy.typing import NDArray
import pandas as pd

__all__ = [
    "get_valid_flux_traces",
    "get_benchmark_flux_traces",
    "create_train_and_test_flux_ts_dataframes",
]

BENCHMARK_FLUX_DATA_FILE_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "data"
    / "flux"
    / "benchmark"
    / "flux_data.json"
)
RAW_FLUX_DATA_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "data"
    / "flux"
    / "raw"
)
print(f"Using flux data path: {RAW_FLUX_DATA_PATH}")
FLUXTRACE_FILENAME_CONVENTION: str = "fluxes_{iteration}.dat"
TIMESERIES_CONVERSION_START_TIMESTAMP: datetime = datetime(2000, 1, 1)
TIMESERIES_CONVERSION_TIMEDELTA_UNIT: str = "ms"


@cache
def load_flux_data(idx: int) -> np.ndarray:
    """Load energy flux data from a .dat file.

    Args:
        idx (int): The id [0, 300] for the flux trace to receive.

    Returns:
        np.ndarray: The energy flux array.
    """
    file_path: Path = RAW_FLUX_DATA_PATH / FLUXTRACE_FILENAME_CONVENTION.format(
        iteration=idx
    )
    data: np.ndarray = np.loadtxt(file_path)
    return data[:, 1]  # only energy flux for now, ignore particle and momentum flux


@cache
def get_valid_flux_traces(full_subsampling: bool = False) -> dict[int, np.ndarray]:
    """Get all valid and subsampled flux traces.

    Args:
        full_subsampling (bool, optional): Whether to use subsampling for all windows. Defaults to False.

    Returns:
        dict[int, np.ndarray]: The dictionary of flux traces.
    """
    nr_flux_traces: int = len(
        list(
            RAW_FLUX_DATA_PATH.glob(FLUXTRACE_FILENAME_CONVENTION.format(iteration="*"))
        )
    )
    print(f"Found {nr_flux_traces} flux traces.")
    HORIZON: int = 240  # head and tail length to consider for mean flux
    SUBSAMPLE_FACTOR: int = 3

    valid_flux_traces: dict[int, np.ndarray] = {}
    incremental_idx: int = 0
    for idx in range(nr_flux_traces):
        flux_data: np.ndarray = load_flux_data(idx)

        # Step 1: Check mean flux at head and tail
        mean_head: float = float(np.mean(flux_data[:HORIZON]))
        mean_tail: float = float(np.mean(flux_data[-HORIZON:]))
        if not (1.0 <= mean_head <= np.inf) or not (1.0 <= mean_tail <= np.inf):
            continue

        # Step 2: Subsample
        subsampled_flux: np.ndarray = flux_data[::SUBSAMPLE_FACTOR]
        if full_subsampling:
            valid_flux_traces[incremental_idx * 3] = subsampled_flux
            valid_flux_traces[(incremental_idx + 1) * 3 - 2] = flux_data[
                1::SUBSAMPLE_FACTOR
            ]
            valid_flux_traces[(incremental_idx + 1) * 3 - 1] = flux_data[
                2::SUBSAMPLE_FACTOR
            ]
        else:
            valid_flux_traces[incremental_idx] = subsampled_flux
        incremental_idx += 1

    return valid_flux_traces


@cache
def create_train_and_test_flux_ts_dataframes(
    n_discretation_quantiles: int = 5,
    test_set_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits flux timeseries data into train and validation sets.
    Splitting is based on stratified sampling of the mean target values.

    Args:
        n_discretation_quantiles (int, optional): In how many bins shall the timeseries means be split for stratified sampling. Defaults to 5.
        test_set_size (float, optional): How big shall the validation set be. Defaults to 0.1.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: training and validation set.
    """
    training_flux_traces: dict[int, np.ndarray] = get_valid_flux_traces(
        full_subsampling=True
    )

    records = []
    for item_id, flux_trace in training_flux_traces.items():
        for t in range(flux_trace.shape[0]):
            records.append(
                {
                    "item_id": item_id,
                    "timestamp": TIMESERIES_CONVERSION_START_TIMESTAMP
                    + pd.to_timedelta(t, unit=TIMESERIES_CONVERSION_TIMEDELTA_UNIT),  # type: ignore
                    "target": flux_trace[t],
                }
            )
    flux_ts_df = pd.DataFrame(records)

    # Compute mean of each series
    mean_df: pd.DataFrame = (
        flux_ts_df.groupby("item_id")["target"].mean().to_frame().reset_index()
    )

    # Bin the target values into quartiles for stratification
    mean_df["target_bin"] = pd.qcut(
        mean_df["target"], q=n_discretation_quantiles, labels=False, duplicates="drop"
    )

    train_ids, val_ids = train_test_split(
        mean_df["item_id"],
        test_size=test_set_size,
        stratify=mean_df["target_bin"],
        random_state=42,
    )
    train_ids: pd.Series
    val_ids: pd.Series

    train_set: pd.DataFrame = flux_ts_df[flux_ts_df["item_id"].isin(train_ids)]
    val_set: pd.DataFrame = flux_ts_df[flux_ts_df["item_id"].isin(val_ids)]

    return train_set, val_set


def get_benchmark_flux_traces() -> dict[
    Literal["ood", "id"], dict[int, NDArray[np.float32]]
]:
    """Get benchmark flux traces for in-distribution and out-of-distribution.

    Returns:
        dict[str, dict[int, NDArray[np.float32]]]: The dictionary of benchmark flux traces with shape (266,).
    """
    flux_data: dict = json.load(open(BENCHMARK_FLUX_DATA_FILE_PATH, "r"))

    benchmark_flux_traces: dict[
        Literal["ood", "id"], dict[int, NDArray[np.float32]]
    ] = {}

    ########################################################
    # In Distribution Benchmark Flux Traces
    ########################################################
    in_distribution_iterations: list[str] = [
        "iteration_8_ifft",
        "iteration_115_ifft",
        "iteration_131_ifft",
        "iteration_148_ifft",
        "iteration_235_ifft",
        "iteration_262_ifft",
    ]
    benchmark_flux_traces["id"] = {
        int(iteration.split("_")[1]): np.array(
            flux_data["in_distribution"][iteration], dtype=np.float32
        )
        for iteration in in_distribution_iterations
    }

    ########################################################
    # Out Of Distribution Benchmark Flux Traces
    ########################################################
    out_of_distribution_iterations: list[str] = [
        "ood_iteration_0_ifft_realpotens",
        "ood_iteration_1_ifft_realpotens",
        "ood_iteration_2_ifft_realpotens",
        "ood_iteration_3_ifft_realpotens",
        "ood_iteration_4_ifft_realpotens",
    ]
    benchmark_flux_traces["ood"] = {
        int(iteration.split("_")[2]): np.array(
            flux_data["out_of_distribution"][iteration], dtype=np.float32
        )
        for iteration in out_of_distribution_iterations
    }

    return benchmark_flux_traces
