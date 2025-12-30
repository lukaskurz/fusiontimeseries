from functools import cache
from pathlib import Path

import numpy as np

__all__ = ["get_valid_flux_traces"]

RAW_FLUX_DATA_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "data"
    / "flux"
    / "raw"
)
print(f"Using flux data path: {RAW_FLUX_DATA_PATH}")
FLUXTRACE_FILENAME_CONVENTION: str = "fluxes_{iteration}.dat"


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


def get_valid_flux_traces() -> dict[int, np.ndarray]:
    """Get all valid and subsampled flux traces.

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
    for idx in range(nr_flux_traces):
        flux_data: np.ndarray = load_flux_data(idx)

        # Step 1: Check mean flux at head and tail
        mean_head: float = float(np.mean(flux_data[:HORIZON]))
        mean_tail: float = float(np.mean(flux_data[-HORIZON:]))
        if not (1.0 <= mean_head <= np.inf) or not (1.0 <= mean_tail <= np.inf):
            continue

        # Step 2: Subsample
        subsampled_flux: np.ndarray = flux_data[::SUBSAMPLE_FACTOR]

        valid_flux_traces[idx] = subsampled_flux

    return valid_flux_traces
