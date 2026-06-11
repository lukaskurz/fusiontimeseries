from typing import Any, Iterator, Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

from fusiontimeseries.lib.benchmarking import rmse_with_standard_error
from fusiontimeseries.lib.conditioning import ConditionRegistry
from fusiontimeseries.lib.dataset import FluxData
from fusiontimeseries.lib.config import FTSConfig
from fusiontimeseries.lib.dataset import FTSDataProcessingMixin

__all__ = ["BaselineTimeseriesDataset", "FTSAblationIterableDataset"]


class FTSDataBenchmarkingMixin:
    @staticmethod
    def evaluate_model_on_test_data(
        model,
        config: FTSConfig,
        benchmark_data: dict[int, FluxData],
        start_context_length: int,
    ) -> dict[str, float | dict[int, list[float]]]:
        forecasts: dict[int, list[float]] = {}
        ground_truths: dict[int, list[float]] = {}
        for flux_id, flux_data in benchmark_data.items():
            flux_data: FluxData
            energy_flux = np.array(flux_data.energy_flux)
            op_params = (
                torch.Tensor(flux_data.operating_parameters)
                .unsqueeze(0)
                .to(config.device)
            )

            ctx: np.ndarray = energy_flux[:start_context_length]
            while len(ctx) < len(energy_flux):
                with torch.no_grad():
                    context_start_idx = config.context_length - len(ctx)
                    tctx = torch.full(
                        size=(1, config.context_length),
                        fill_value=config.padding_value,
                    )
                    tctx[0, context_start_idx:] = torch.tensor(ctx)
                    context_mask = torch.full_like(
                        tctx, fill_value=config.padding_mask_default
                    )  # 0.0
                    # context_mask[:cutoff_idx] = self.config.padding_mask_indicator context_masks.append(context_mask)
                    context_mask[0, :context_start_idx] = (
                        config.padding_mask_indicator
                    )  # 1.0

                    with ConditionRegistry.patch(op_params=op_params):
                        with torch.autocast(
                            device_type=config.device, dtype=torch.float32
                        ):
                            # predictions shape: (batch_size, num_patches, horizon_len, num_quantiles + 1(mean))
                            predictions: torch.Tensor = model(
                                tctx.to(config.device, non_blocking=True),
                                context_mask.to(
                                    config.device, non_blocking=True
                                ).float(),
                                torch.tensor([[0.0]], dtype=torch.long).to(
                                    config.device, non_blocking=True
                                ),
                            )
                forecast = (
                    predictions[0, -1, : config.prediction_length, 0].cpu().numpy()
                )
                ctx = np.concatenate([ctx, forecast])

            forecasts[flux_id] = ctx[: len(energy_flux)].tolist()
            ground_truths[flux_id] = flux_data.energy_flux

        benchmark_means: list[np.floating] = [
            np.mean(gt[-config.pred_tail_timestamps :]) for gt in ground_truths.values()
        ]
        forecast_means: list[np.floating] = [
            np.mean(fc[-config.pred_tail_timestamps :]) for fc in forecasts.values()
        ]
        rmse, rmse_se = rmse_with_standard_error(
            y_true=np.array(benchmark_means), y_pred=np.array(forecast_means)
        )
        return {
            "rmse": rmse,
            "rmse_standard_error": rmse_se,
            "forecasts": forecasts,
            "ground_truths": ground_truths,
        }


class BaselineTimeseriesDataset(
    Dataset, FTSDataProcessingMixin, FTSDataBenchmarkingMixin
):
    def __init__(
        self,
        time_series: list[np.ndarray],
        config: FTSConfig,
        operating_parameters: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.time_series = time_series
        self.config = config
        self.operating_parameters = operating_parameters
        self.samples: list[
            tuple[int, int]
        ] = []  # list of (time_series_idx, cutoff_idx)
        for idx in range(len(time_series)):
            for cutoff_idx in self.config.train_context_cutoffs:
                self.samples.append((idx, cutoff_idx))

        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def prepare_sample(self, idx: int) -> dict[str, torch.Tensor]:
        timeseries_index, cutoff_idx = self.samples[idx]
        time_series: np.ndarray = self.time_series[timeseries_index]

        ############# Context ##############
        history: np.ndarray = time_series[:cutoff_idx, ...]
        if cutoff_idx <= 1:
            history = np.zeros_like(history)
        context_start_idx = self.config.context_length - cutoff_idx

        context = torch.full(
            size=(self.config.context_length,),
            fill_value=self.config.padding_value,
            dtype=torch.float32,
        )
        context[context_start_idx:] = torch.Tensor(history)

        ############## Context mask ##############
        context_mask = torch.full_like(
            context,
            fill_value=self.config.padding_mask_default,
            dtype=torch.float32,
        )
        # assign padding indicator to padded positions (opposite to chronos2)
        context_mask[:context_start_idx] = self.config.padding_mask_indicator

        ############### Future ##############
        target: np.ndarray = time_series[
            cutoff_idx : cutoff_idx + self.config.prediction_length, ...
        ]
        future_target = torch.tensor(target, dtype=torch.float32)

        ############## Frequency ##############
        freq = torch.tensor([0.0], dtype=torch.long)

        sample = {
            "context": context,
            "context_mask": context_mask,
            "future_target": future_target,
            "freq": freq,
        }
        if self.operating_parameters is not None:
            sample["operating_parameters"] = torch.Tensor(
                self.operating_parameters[timeseries_index]
            )
        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.prepare_sample(idx)


class FTSAblationIterableDataset(
    IterableDataset, FTSDataProcessingMixin, FTSDataBenchmarkingMixin
):
    def __init__(
        self,
        time_series: list[np.ndarray],
        config: FTSConfig,
        mode: Literal["train", "val"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.time_series = time_series
        self.config = config
        self.mode = mode

        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        if self.config.sampling_strategy == "tail_mean":
            self.prepare_tail_mean_stratified_sampling()

    def prepare_tail_mean_stratified_sampling(self):
        # assign each time series to a bin based on mean energy flux in the tail
        self.bin_assignments: list[np.ndarray] = []
        tail_means: list[np.floating] = [
            np.mean(time_series[-self.config.pred_tail_timestamps :])
            for time_series in self.time_series
        ]
        bin_assignment = pd.qcut(
            tail_means,  # type: ignore
            q=self.config.sampling_bins,
            labels=False,
            duplicates="drop",
        )  # type: ignore
        print(f"Tail mean bins: {np.unique(bin_assignment, return_counts=True)}")
        self.bin_assignments.append(bin_assignment)

    # @staticmethod
    # def collate_fn(
    #     batch: list[dict[str, torch.Tensor]],
    # ) -> dict[str, list[torch.Tensor]]:
    #     collated_batch = {}
    #     for key in batch[0].keys():
    #         new_value = torch.stack([sample[key] for sample in batch])
    #         # print(f"Collating key: {key}, shape: {new_value.shape}")
    #         collated_batch[key] = new_value
    #     return collated_batch

    # def prepare_sample(self, idx: int) -> dict[str, torch.Tensor]:
    #     time_series: np.ndarray = self.time_series[idx]
    #     L: int = len(time_series)
    #     cutoff_idx: int = np.random.randint(80, L - self.config.prediction_length + 1)

    #     ############# Context ##############
    #     history: np.ndarray = time_series[:cutoff_idx, ...]
    #     context_start_idx = self.config.context_length - cutoff_idx

    #     context = torch.full(
    #         size=(self.config.context_length,),
    #         fill_value=self.config.padding_value,
    #         dtype=torch.float32,
    #     )
    #     context[context_start_idx:] = torch.Tensor(history)

    #     ############## Context mask ##############
    #     context_mask = torch.full_like(
    #         context,
    #         fill_value=self.config.padding_mask_default,
    #         dtype=torch.float32,
    #     )
    #     # assign padding indicator to padded positions (opposite to chronos2)
    #     context_mask[:context_start_idx] = self.config.padding_mask_indicator

    #     ############### Future ##############
    #     target: np.ndarray = time_series[
    #         cutoff_idx : cutoff_idx + self.config.prediction_length, ...
    #     ]
    #     future = torch.tensor(target, dtype=torch.float32)

    #     ############## Frequency ##############
    #     freq = torch.tensor([0.0], dtype=torch.long)

    #     return {
    #         "context": context,
    #         "context_mask": context_mask,
    #         "future_target": future,
    #         "freq": freq,
    #     }

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        collated_batch = {}
        for key in batch[0].keys():
            new_value = torch.cat([sample[key] for sample in batch])
            # print(f"Collating key: {key}, shape: {new_value.shape}")
            collated_batch[key] = new_value
        return collated_batch

    def prepare_sample(self, idx: int) -> dict[str, torch.Tensor]:
        time_series: np.ndarray = self.time_series[idx]

        contexts: list[torch.Tensor] = []
        context_masks: list[torch.Tensor] = []
        future_targets: list[torch.Tensor] = []
        freqs: list[torch.Tensor] = []

        for cutoff_idx in self.config.train_context_cutoffs:
            ############# Context ##############
            history: np.ndarray = time_series[:cutoff_idx, ...]
            context_start_idx = self.config.context_length - cutoff_idx

            context = torch.full(
                size=(self.config.context_length,),
                fill_value=self.config.padding_value,
                dtype=torch.float32,
            )
            context[context_start_idx:] = torch.Tensor(history)
            contexts.append(context)

            ############## Context mask ##############
            context_mask = torch.full_like(
                context,
                fill_value=self.config.padding_mask_default,
                dtype=torch.float32,
            )
            # assign padding indicator to padded positions (opposite to chronos2)
            context_mask[:context_start_idx] = self.config.padding_mask_indicator
            context_masks.append(context_mask)

            ############### Future ##############
            target: np.ndarray = time_series[
                cutoff_idx : cutoff_idx + self.config.prediction_length, ...
            ]
            future = torch.tensor(target, dtype=torch.float32)
            future_targets.append(future)

            ############## Frequency ##############
            freqs.append(torch.tensor([0.0], dtype=torch.long))

        return {
            "context": torch.stack(contexts),
            "context_mask": torch.stack(context_masks),
            "future_target": torch.stack(future_targets),
            "freq": torch.stack(freqs).to(torch.long),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.mode == "train":
            while True:
                # indefinitely iterate and yield random samples, stratified by bins
                for bin_assignment in self.bin_assignments:
                    unique_bins = np.unique(bin_assignment)
                    for selected_bin in unique_bins:
                        # a list of time series indices that belong to the selected bin
                        candidates = np.where(bin_assignment == selected_bin)[0]
                        if len(candidates) == 0:
                            continue

                        time_series_idx: int = np.random.choice(candidates)
                        yield self.prepare_sample(time_series_idx)
        else:
            # in validation mode, we want deterministic iteration over the dataset
            for idx in range(len(self.time_series)):
                yield self.prepare_sample(idx)
