"""
Embedding module from jku-iml/neural-gyrokinetics
"""

from functools import partial
from einops import rearrange
import torch
from torch import nn


__all__ = ["ContinuousConditionEmbed"]


def _seq_weight_init(weight_init_fn, bias_init_fn=None):
    if bias_init_fn is None:
        bias_init_fn = nn.init.zeros_

    def _apply(m):
        if isinstance(m, nn.Linear):
            weight_init_fn(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                bias_init_fn(m.bias)

    return _apply


class ContinuousConditionEmbed(nn.Module):
    omega: torch.Tensor

    def __init__(
        self,
        embedding_dim: int,
        n_cond: int,
        max_wavelength: int = 10_000,
        init_weights: str | None = "kaiming_uniform",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_cond = n_cond
        self.ndim_padding = self.embedding_dim % n_cond
        dim_per_ndim = (self.embedding_dim - self.ndim_padding) // n_cond
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * n_cond
        cond_per_wave = (self.embedding_dim - self.padding) // n_cond
        assert cond_per_wave > 0
        self.register_buffer(
            "omega",
            1.0 / max_wavelength ** (torch.arange(0, cond_per_wave, 2) / cond_per_wave),
        )
        self.cond_dim = embedding_dim

        # prefix lora_ to set trainable using loralib
        self.lora_opc_embed = nn.Sequential(
            nn.Linear(embedding_dim, self.cond_dim),
            nn.SiLU(),
        )

        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.lora_opc_embed.apply(_seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights == "kaiming_uniform":
            self.lora_opc_embed.apply(
                _seq_weight_init(
                    partial(
                        nn.init.kaiming_uniform_,
                        nonlinearity="relu",
                        mode="fan_in",
                        a=0,
                    )
                )
            )
        elif init_weights == "normal_smallvar":
            self.lora_opc_embed.apply(
                _seq_weight_init(partial(nn.init.normal_, mean=0.0, std=1e-3))
            )
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.lora_opc_embed.apply(_seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        assert self.n_cond == cond.shape[-1], f"{self.n_cond} != {cond.shape[-1]}"

        # (b n cdim/2) = (b n 1) x (1 cdim/2)
        out = cond.unsqueeze(-1) @ self.omega.unsqueeze(0)

        # (b n cdim/2) stacked at cdim: (b n 2w)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        emb = rearrange(emb, "... ncond cdim -> ... (ncond cdim)")

        if self.padding > 0:
            padding = torch.zeros(
                *emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype
            )
            emb = torch.concat([emb, padding], dim=-1)

        emb = self.lora_opc_embed(emb)
        return emb
