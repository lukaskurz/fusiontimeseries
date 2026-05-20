import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

from torch.utils.data import IterableDataset


from fusiontimeseries.lib.conditioning import ConditionRegistry
from fusiontimeseries.lib.config import FTSConfig


__all__ = ["TimesFMTrainer"]


class TimesFMTrainer(Trainer):
    def __init__(
        self,
        model: PatchedTimeSeriesDecoder,
        train_args: TrainingArguments,
        train_dataset: IterableDataset,
        eval_dataset: IterableDataset,
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

    def _quantile_loss(
        self, pred: torch.Tensor, actual: torch.Tensor, quantile: float
    ) -> torch.Tensor:
        """Calculates quantile loss.
        Args:
            pred: Predicted values
            actual: Actual values
            quantile: Quantile at which loss is computed
        Returns:
            Quantile loss
        """
        dev = actual - pred
        loss_first = dev * quantile
        loss_second = -dev * (1.0 - quantile)
        return 2 * torch.where(loss_first >= 0, loss_first, loss_second)

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
            # predictions shape: (batch_size, num_patches, prediction_length, mean + num_quantiles)
            predictions: torch.Tensor = model(
                input_ts=input_ts, input_padding=input_padding, freq=freq
            )

        quantile_losses = []
        for i, quantile in enumerate(model.config.quantiles):
            last_patch_quantile = predictions[
                :, -1, : self.fts_config.prediction_length, i + 1
            ]
            quantile_loss = torch.mean(
                self._quantile_loss(
                    last_patch_quantile, target_ts.squeeze(-1), quantile
                )
            )
            quantile_losses.append(quantile_loss)

        # target_ts (batch_size, pred_len)
        # predictions (batch_size, num_patches, pred_len, 1 + num_quantiles)
        mse_loss = F.mse_loss(predictions[:, -1, :, 0], target_ts)

        loss = torch.mean(torch.stack(quantile_losses)) + mse_loss
        return (loss, predictions) if return_outputs else loss
