import json
from pathlib import Path
from typing import Any, Literal
import numpy as np
import torch
from torch.utils.data import Dataset

from fusiontimeseries.experiments.config import FinetuningConfig

type Flux = dict[str | Literal["energy_flux"], Any]
# Hierarchy: namespace -> simulation -> flux data
type SimulationID = int
type Namespace = str
type FluxData = dict[Namespace, dict[SimulationID, Flux]]

flux_data_path: Path = (
    Path(__file__).parent.parent.parent.parent / "data" / "flux_data.json"
)
print(flux_data_path)


class FTSDataProcessingMixin:
    @staticmethod
    def subsample_flux(flux: Flux, subsample_factor: int) -> Flux:
        energy_flux = flux["energy_flux"]
        return {
            **flux,
            "energy_flux": energy_flux[::subsample_factor],
        }


def prepare_model_input(
    flux: Flux,
    cutoff_idx: int,
    config: FinetuningConfig,
    history_override: np.ndarray | None = None,
) -> dict[
    Literal["context", "context_mask", "future_target", "freq", "operating_parameters"],
    torch.Tensor,
]:
    time_series = np.array(flux["energy_flux"])
    ############# Context ##############
    if history_override is not None:
        history = history_override
    else:
        history = time_series[:cutoff_idx]

    if cutoff_idx <= 1:
        history = np.zeros(
            (1), dtype=np.float32
        )  # Ensure at least one value in history for very small cutoff_idx
    context_start_idx = config.context_length - cutoff_idx

    context = torch.full(
        size=(config.context_length,),
        fill_value=config.padding_value,
        dtype=torch.float32,
    )
    context[context_start_idx:] = torch.Tensor(history)

    ############## Context mask ##############
    context_mask = torch.full_like(
        context,
        fill_value=config.padding_mask_default,
        dtype=torch.float32,
    )
    # assign padding indicator to padded positions (opposite to chronos2)
    context_mask[:context_start_idx] = config.padding_mask_indicator

    ############### Future ##############
    target: np.ndarray = time_series[
        cutoff_idx : cutoff_idx + config.prediction_length, ...
    ]
    future_target = torch.tensor(target, dtype=torch.float32)

    ############## Frequency ##############
    freq = torch.tensor([0.0], dtype=torch.long)

    ############### Operating parameters ##############
    operating_parameters = torch.tensor(
        [flux["q"], flux["shat"], flux["rlt"], flux["rln"]],
        dtype=torch.float32,
    )

    sample = {
        "context": context,
        "context_mask": context_mask,
        "future_target": future_target,
        "freq": freq,
        "operating_parameters": operating_parameters,
    }
    return sample  # type: ignore


class FluxDataset(Dataset, FTSDataProcessingMixin):
    def __init__(
        self,
        namespaces: list[
            Literal[
                "batch_1",
                "batch_2",
                "batch_3",
                "batch_4",
                "batch_5",
                "batch_6",
                "batch_7",
                "batch_8",
                "batch_9",
                "batch_10",
                "gyroswin_id",
                "gyroswin_ood",
                "gyroswin_train",
                "gyroswin_val",
            ]
            | str
        ],
        config: FinetuningConfig,
    ) -> None:
        self.namespaces = namespaces
        self.config = config

        self.flux_data: FluxData = {}
        with open(flux_data_path, "r") as f:
            all_flux_data = json.load(f)
            for ns in namespaces:
                namespace_data: dict[SimulationID, Flux] = all_flux_data[ns]
                if self.config.subsampling:
                    for sim_id, flux in namespace_data.items():
                        subsampled_flux = {
                            **flux,
                            "energy_flux": flux["energy_flux"][::3],
                        }
                        namespace_data[sim_id] = subsampled_flux
                self.flux_data[ns] = namespace_data
            del all_flux_data

        self.samples: list[tuple[Namespace, SimulationID, int]] = []
        for namespace, flux_data in self.flux_data.items():
            for simulation_id in flux_data.keys():
                for cutoff_idx in self.config.train_context_cutoffs:
                    self.samples.append((namespace, simulation_id, cutoff_idx))

        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def prepare_sample(self, idx: int) -> dict[str, torch.Tensor]:
        namespace, simulation_idx, cutoff_idx = self.samples[idx]
        flux: Flux = self.flux_data[namespace][simulation_idx]
        sample = prepare_model_input(flux, cutoff_idx, self.config)
        return sample  # type: ignore

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.prepare_sample(idx)
