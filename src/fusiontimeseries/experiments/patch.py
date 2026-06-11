import torch
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from timesfm.pytorch_patched_decoder import TimesFMConfig
from fusiontimeseries.ablations.trainer import patch_times_fm_config


def _no_norm_forward_transform(
    self, inputs: torch.Tensor, patched_pads: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Input is of shape [B, N, P]."""
    mu = torch.tensor([0.0], device=inputs.device)
    sigma = torch.tensor([1.0], device=inputs.device)
    outputs = inputs  # no normalization
    return outputs, (mu, sigma)


def _no_norm_reverse_transform(
    self, outputs: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    return outputs  # no normalization, so reverse is identity


def apply_patches():

    # patch the model's normalization functions to no-op for the base ablation
    PatchedTimeSeriesDecoder._forward_transform = _no_norm_forward_transform
    PatchedTimeSeriesDecoder._reverse_transform = _no_norm_reverse_transform

    # patch config to be HF Trainer compatible
    patch_times_fm_config(TimesFMConfig)
