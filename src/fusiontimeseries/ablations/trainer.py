from typing import Type
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

from torch.utils.data import Dataset

from timesfm.pytorch_patched_decoder import TimesFMConfig

from fusiontimeseries.lib.conditioning import ConditionRegistry
from fusiontimeseries.lib.config import FTSConfig


__all__ = ["TimesFMTrainer", "patch_times_fm_config"]


def patch_times_fm_config(TimesFMConfig: Type[TimesFMConfig]):
    def get(self, attr_name: str, default=None):
        return getattr(self, attr_name, default)

    TimesFMConfig.get = get  # type: ignore

    # add __iter method to TimesFMConfig to avoid error on save_pretrained: TimesFMConfig object is not iterable
    def __iter__(self):
        for key in self.__dict__:
            yield key, self.__dict__[key]

    TimesFMConfig.__iter__ = __iter__  # type: ignore


class TimesFMTrainer(Trainer):
    def __init__(
        self,
        model: PatchedTimeSeriesDecoder,
        train_args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        fts_config: FTSConfig,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            *args,
            **kwargs,
        )
        self.fts_config = fts_config
        self.model: PatchedTimeSeriesDecoder

    def _quantile_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # predictions shape: (batch_size, pred_len, num_quantiles)
        # targets shape: (batch_size, pred_len)

        quantiles = self.model.config.quantiles
        losses = []
        for i, q in enumerate(quantiles):
            errors = targets - predictions[..., i]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
        return torch.stack(losses).mean()

    def compute_loss(
        self,
        model: PatchedTimeSeriesDecoder,
        inputs: dict[str, torch.Tensor],
        *args,
        return_outputs=False,
        **kwargs,
    ):
        # Tensor[B, N]
        p_raw: torch.Tensor | None = inputs.pop(
            "operating_parameters", None
        )  # remove before forward, otherwise TypeError in Trainer

        input_ts: torch.Tensor = inputs["context"]
        target_ts: torch.Tensor = inputs["future_target"]
        input_padding: torch.Tensor = inputs["context_mask"]
        freq: torch.Tensor = inputs["freq"]

        with ConditionRegistry.patch(op_params=p_raw):
            # predictions shape: (batch_size, context_len // input_patch_size, prediction_length, mean + num_quantiles)
            predictions: torch.Tensor = model(
                input_ts=input_ts, input_padding=input_padding, freq=freq
            )

        # target_ts (batch_size, pred_len)
        # predictions (batch_size, num_patches, pred_len, 1 + num_quantiles)
        huber_loss = F.smooth_l1_loss(
            predictions[:, -1, : self.fts_config.prediction_length, 0], target_ts
        )
        quantile_loss = self._quantile_loss(
            predictions[
                :, -1, : self.fts_config.prediction_length, 1:
            ],  # take the quantile predictions of the last patch
            target_ts,
        )
        print(
            f"Huber Loss: {huber_loss.item():.4f}, Quantile Loss: {quantile_loss.item():.4f}"
        )

        loss = quantile_loss + huber_loss
        return (loss, predictions) if return_outputs else loss
