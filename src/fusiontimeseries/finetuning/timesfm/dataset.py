import numpy as np
import torch
from fusiontimeseries.lib.config import FTSConfig
from fusiontimeseries.lib.dataset import TimeseriesDataset


__all__ = ["TimesFMDataset"]


class TimesFMDataset(TimeseriesDataset):
    config: FTSConfig

    def prepare_sample(self, idx: int) -> dict[str, torch.Tensor]:
        time_series: np.ndarray = self.time_series[idx]
        operating_parameters: np.ndarray = self.ops[idx, ...]

        L: int = len(time_series)
        cutoff_idx: int = np.random.randint(
            self.config.prediction_length, L - self.config.prediction_length + 1
        )

        # Context
        history: np.ndarray = time_series[:cutoff_idx, ...]
        augmented_history, augmented_ops = self.apply_data_augmentation(
            history, operating_parameters
        )
        context = torch.full(
            size=(self.config.context_length,), fill_value=self.config.padding_value
        )
        context[-cutoff_idx:] = torch.Tensor(augmented_history)
        # looks like [0, 0, 0, ..., 0, val, val, val, ..., val]

        # Context mask
        context_mask = torch.full_like(
            context, fill_value=self.config.padding_mask_default
        )
        # assign padding indicator to padded positions (opposite to chronos2)
        context_mask[:cutoff_idx] = self.config.padding_mask_indicator

        # Future and Future mask
        target: np.ndarray = time_series[
            cutoff_idx : cutoff_idx + self.config.prediction_length, ...
        ]
        future = torch.Tensor(target)

        return {
            "context": context,
            "context_mask": context_mask,
            "future_target": future,
            "freq": torch.tensor([1.0], dtype=torch.long),
            "operating_parameters": torch.Tensor(augmented_ops),
        }
