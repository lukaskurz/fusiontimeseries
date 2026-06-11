#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from transformers import Conv1D

import math
from typing import List, Any


from fusiontimeseries.lib.conditioning import ConditionRegistry
from fusiontimeseries.lib.modules import ContinuousConditionEmbed


def expand_like(target: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Expands the target tensor to have the same shape as the like tensor,
    by adding singleton dimensions where necessary.

    Args:
        target (torch.Tensor): (B, ..., N), The tensor to be expanded.
        like (torch.Tensor): (..., B, ..., N, ...), The tensor whose shape is to be matched.

    Returns:
        torch.Tensor: The expanded tensor.
    """
    assert target.ndim >= 2, (
        "Target tensor must have at least 2 dimensions (batch and feature)"
    )
    assert target.ndim <= like.ndim, (
        "Target tensor cannot have more dimensions than 'like' tensor"
    )

    batch_size = target.shape[0]
    feature_size = target.shape[-1]

    batch_dim_in_like = (torch.tensor(like.shape) == batch_size).nonzero(as_tuple=True)[
        0
    ]

    if len(batch_dim_in_like) == 0:
        raise RuntimeError(
            f"Could not find batch size {batch_size} in tensor 'like' of shape {like.shape}"
        )

    # 2. Create a view of target that aligns with 'like'
    # We want target to be 1s everywhere except the batch dim and the feature dim
    new_shape = [1] * like.ndim
    new_shape[batch_dim_in_like[0]] = batch_size
    new_shape[-1] = feature_size

    exp_target = target.view(*new_shape)
    return exp_target


class LoRALayer:
    def __init__(
        self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

    def gram_schmidt(self, vv):
        def projection(u, v):
            return (v.T @ u) / (u.T @ u) * u

        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)
        uu[:, 0] = vv[:, 0].clone()
        for k in range(1, nk):
            vk = vv[:, k].clone()
            uk = 0
            for j in range(0, k):
                uj = uu[:, j].clone()
                uk = uk + projection(uj, vk)
            uu[:, k] = vk - uk
        uu = uu / torch.linalg.norm(uu, axis=0, keepdims=True)
        return uu

    def householder_transform(self, A):
        """
        returns a semi-orthogonal matrix computed via householder transform
        """
        m, n = A.data.shape
        mat = A.data
        if n > m:
            h, tau = torch.geqrf(mat.T)
            q = torch.linalg.householder_product(h, tau).T
        else:
            h, tau = torch.geqrf(mat)
            q = torch.linalg.householder_product(h, tau)
        return q

    @classmethod
    def convert(
        cls,
        module: nn.Module,
        kind: str = "LoRA",
        lora_rank=0,
        lora_alpha=1,
        # names_to_exclude=None,
        target_module_names: list[str] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ):
        assert kind in layer_dict, (
            f"unknown LoRA layer kind {kind}, Possible choices: [LoRA, VeRA, DoRA, LoRAM]"
        )
        module_output = module
        if isinstance(module, nn.Linear):
            if target_module_names is None or (
                name is not None
                and any([target_name in name for target_name in target_module_names])
            ):
                # print(f"Converting module: {name} to LoRA layer")
                module_output = layer_dict[kind](
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    **kwargs,
                )
                module_output.weight = module.weight
                module_output.bias = module.bias
        elif isinstance(module, Conv1D):
            if target_module_names is None or name in target_module_names:
                module_output = layer_dict[kind](
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1],
                    bias=True,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    fan_in_fan_out=True,
                    **kwargs,
                )
                module_output.weight = module.weight
                module_output.bias = module.bias
        for child_name, child in module.named_children():
            module_output.add_module(
                child_name,
                cls.convert(
                    module=child,
                    kind=kind,
                    lora_rank=lora_rank,
                    target_module_names=target_module_names,
                    name=child_name if name is None else f"{name}.{child_name}",
                    **kwargs,
                ),
            )
        del module
        return module_output


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        initialize: bool = False,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
        self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_B)
            nn.init.normal_(self.lora_A)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x,
                self.lora_A.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    # Class-level shared projection (will be set once for all instances)
    _shared_p_projection: ContinuousConditionEmbed | None = None

    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        post_layer_norm: bool = False,
        pre_batch_norm: bool = False,
        **kwargs,
    ):
        self.dtype = kwargs.get("dtype", torch.float32)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features
        self.post_layer_norm = post_layer_norm
        self.pre_batch_norm = pre_batch_norm
        self._init_lora(r)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        if self.post_layer_norm:
            self.post_ln = nn.LayerNorm(out_features)
            self.merge_weights = False
        if self.pre_batch_norm:
            self.pre_bn = nn.BatchNorm1d(in_features, affine=False)
            self.merge_weights = False

    def _init_lora(self, r):
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, self.in_features))).to(
                self.dtype
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((self.out_features, r))
            ).to(self.dtype)
            self.scaling = self.lora_alpha / r
        else:
            try:
                # ensure parameters do not exist if they are zero
                delattr(self, "lora_A")
                delattr(self, "lora_B")
                delattr(self, "scaling")
            except AttributeError:
                pass
        self.weight.requires_grad = False
        self.r = r

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # adapt to initialization via PCA on pretrained weights
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.pre_batch_norm:
                x = self.pre_bn(x)
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            if self.post_layer_norm:
                result = self.post_ln(result)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    def change_lora_rank(self, new_rank):
        if new_rank != self.r:
            self._init_lora(new_rank)


class DoRALinear(nn.Linear, LoRALayer):
    # DoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=False,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))

            # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def set_weight_norm(self):
        self.m.data.copy_(self.weight.norm(p=2, dim=0, keepdim=True))

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # adapt to initialization via PCA on pretrained weights
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        lora = self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1) * self.scaling
        combined_weight = self.weight + lora.T
        column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
        V = combined_weight / column_norm
        new_weight = self.m * V
        result = F.linear(x, T(new_weight), bias=self.bias)
        return result

    def change_lora_rank(self, new_rank):
        if new_rank != self.r:
            self._init_lora(new_rank)  # type: ignore


class VeRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=r, lora_alpha=1, lora_dropout=lora_dropout, merge_weights=False
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            std_dev = 1 / torch.sqrt(torch.tensor(r).float())
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * std_dev)
            self.lora_B = nn.Parameter(torch.randn(out_features, r) * std_dev)
            self.scale_b = nn.Parameter(torch.zeros((1, out_features)))
            self.scale_a = nn.Parameter(torch.ones((1, r)))

            # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        lora = (
            self.lora_A.transpose(0, 1)
            * self.scale_a
            @ self.lora_B.transpose(0, 1)
            * self.scale_b
        )
        combined_weight = self.weight + lora.T
        result = F.linear(x, T(combined_weight), bias=self.bias)
        return result


class EVALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=r, lora_alpha=1, lora_dropout=lora_dropout, merge_weights=False
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # init lora_A at random, but will be updated with PCA weights upon execution of training.
            self.lora_A = nn.Parameter(torch.zeros(in_features, in_features))
            self.lora_B = nn.Parameter(torch.zeros(in_features, out_features))
            self.singvals = nn.Parameter(torch.ones((1, in_features)))
            # self.scale_pre_trained = nn.Parameter((torch.ones(1, out_features)))

            # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        lora = (self.lora_A.transpose(0, 1) * self.scale_a) @ self.lora_B.transpose(  # type: ignore
            0, 1
        )
        combined_weight = self.weight * self.scale_pre_trained + lora.T  # type: ignore
        result = F.linear(x, T(combined_weight), bias=self.bias)
        return result


class NormalizedWeight(nn.Module):
    def __init__(self, weight=None, m=None):
        super().__init__()
        if weight is not None:
            m = weight.norm(p=2, dim=0, keepdim=True)
        self.register_parameter("m", nn.Parameter(m))  # type: ignore

    def forward(self, X):
        return X * self.m


class LoRAMLinear(Linear):
    # LoRA with a trainable magnitude vector

    def register_weight_parametrization(self):
        # parametrize pretrained weights with trainable scaling vector and freeze+noramlize original weights
        m = self.weight.norm(p=2, dim=0, keepdim=True)
        self.weight = nn.Parameter(self.weight / m)
        self.weight.requires_grad = False
        parametrize.register_parametrization(self, "weight", NormalizedWeight(m=m))

    def remove_weight_parametrization(self):
        parametrize.remove_parametrizations(self, "weight", leave_parametrized=True)
        self.weight.requires_grad = True


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert out_features % len(enable_lora) == 0, (
            "The length of enable_lora must divide out_features"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self._init_lora(r)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def _init_lora(self, r):
        # Actual trainable parameters
        if r > 0 and any(self.enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(self.enable_lora), self.in_features))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (
                        self.out_features
                        // len(self.enable_lora)
                        * sum(self.enable_lora),
                        r,
                    )
                )
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / r
            # Freezing the pre-trained weight matrix
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (self.out_features,), dtype=torch.bool
            ).view(len(self.enable_lora), -1)
            self.lora_ind[self.enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        else:
            try:
                # ensure parameters do not exist if they are zero
                delattr(self, "lora_A")
                delattr(self, "lora_B")
                delattr(self, "scaling")
                delattr(self, "lora_ind")
            except AttributeError:
                pass
        self.weight.requires_grad = False
        self.r = r

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora),
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

    def change_lora_rank(self, new_rank):
        if new_rank != self.r:
            self._init_lora(new_rank)


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(
        self,
        conv_module,
        in_channels,
        out_channels,
        kernel_size,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=True,
        **kwargs,
    ):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros(
                    (out_channels // self.conv.groups * kernel_size, r * kernel_size)
                )
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
        self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(
                        self.conv.weight.shape
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(
                        self.conv.weight.shape
                    ) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight
                + (self.lora_B @ self.lora_A).view(self.conv.weight.shape)
                * self.scaling,
                self.conv.bias,
            )
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this


class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)


# Operating Parameter Conditioning Adapters

OP_PARAM_KEY: str = "op_params"


class BilinearLoRA(nn.Linear, LoRALayer):
    # Class-level shared projection (will be set once for all instances)
    _shared_p_projection: ContinuousConditionEmbed | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,  # keep false due to conditioning
        post_layer_norm: bool = False,
        pre_batch_norm: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features
        self.post_layer_norm = post_layer_norm
        self.pre_batch_norm = pre_batch_norm

        self.lora_scale = self.lora_alpha / r
        self.p_dim = 512

        self._init_lora(r)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        if self.post_layer_norm:
            self.post_ln = nn.LayerNorm(out_features)
            self.merge_weights = False
        if self.pre_batch_norm:
            self.pre_bn = nn.BatchNorm1d(in_features, affine=False)
            self.merge_weights = False

    def _init_lora(self, r):
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, self.in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, r)))
            self.lora_C = nn.Parameter(self.weight.new_zeros((r, self.p_dim)))
        else:
            try:
                # ensure parameters do not exist if they are zero
                delattr(self, "lora_A")
                delattr(self, "lora_B")
                delattr(self, "lora_C")
                delattr(self, "lora_scale")
            except AttributeError:
                pass
        self.weight.requires_grad = False
        self.r = r

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # initialize A the same way as the default for nn.Linear and B to zero
        # adapt to initialization via PCA on pretrained weights
        if hasattr(self, "lora_C"):
            nn.init.kaiming_uniform_(self.lora_C, a=math.sqrt(5))
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if hasattr(self, "lora_B"):
            nn.init.zeros_(self.lora_B)

    def change_lora_rank(self, new_rank):
        if new_rank != self.r:
            self._init_lora(new_rank)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.lora_scale
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.lora_scale
                self.merged = True

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, ..., in_features)
        """

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # Base linear projection
        y_base = F.linear(x, T(self.weight), bias=self.bias)  # (B, ..., out_features)
        if self.pre_batch_norm:
            x = self.pre_bn(x)

        if self.r > 0 and not self.merged:
            p: torch.Tensor | None = ConditionRegistry.get(OP_PARAM_KEY)
            if p is not None and self._shared_p_projection is not None:
                p = p.to(self.lora_C.device)

                # Learnable representation of p: (B, p_dim)
                p_repr: torch.Tensor = self._shared_p_projection(p)

                # condition projection: (B, r) = (B, p_dim) @ (r, p_dim).T
                c = F.linear(p_repr, self.lora_C)

                # input projection: (B, ..., r) = ( (B, ..., in_features) @ (r, in_features).T )
                h = F.linear(self.lora_dropout(x), self.lora_A)

                # Bilinear modulation in rank space
                h_mod = h * (1.0 + expand_like(target=c, like=h))

                # Project back to output space
                delta_y = (
                    F.linear(h_mod, self.lora_B) * self.lora_scale
                )  # (B, ..., out_features)
                y_base += delta_y

            else:
                raise RuntimeError(
                    "Operating parameters not found or _shared_p_projection is None during forward pass."
                )

            if self.post_layer_norm:
                y_base = self.post_ln(y_base)

        return y_base


class RSSBilinearLoRA(nn.Linear, LoRALayer):
    # Class-level shared projection (will be set once for all instances)
    _shared_p_projection: ContinuousConditionEmbed | None = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,  # keep false due to conditioning
        post_layer_norm: bool = False,
        pre_batch_norm: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features
        self.post_layer_norm = post_layer_norm
        self.pre_batch_norm = pre_batch_norm

        self.lora_scale = self.lora_alpha / r
        self.p_dim = 512

        self._init_lora(r)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        if self.post_layer_norm:
            self.post_ln = nn.LayerNorm(out_features)
            self.merge_weights = False
        if self.pre_batch_norm:
            self.pre_bn = nn.BatchNorm1d(in_features, affine=False)
            self.merge_weights = False

    def _init_lora(self, r):
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, self.in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, r)))
            self.lora_C = nn.Parameter(self.weight.new_zeros((r, self.p_dim)))
            self.lora_S = nn.Parameter(self.weight.new_zeros((r, self.p_dim)))
        else:
            try:
                # ensure parameters do not exist if they are zero
                delattr(self, "lora_A")
                delattr(self, "lora_B")
                delattr(self, "lora_C")
                delattr(self, "lora_S")
                delattr(self, "lora_scale")
            except AttributeError:
                pass
        self.weight.requires_grad = False
        self.r = r

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # initialize A the same way as the default for nn.Linear and B to zero
        # adapt to initialization via PCA on pretrained weights
        if hasattr(self, "lora_C"):
            nn.init.kaiming_uniform_(self.lora_C, a=math.sqrt(5))
        if hasattr(self, "lora_S"):
            nn.init.zeros_(self.lora_S)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if hasattr(self, "lora_B"):
            nn.init.zeros_(self.lora_B)

    def change_lora_rank(self, new_rank):
        if new_rank != self.r:
            self._init_lora(new_rank)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.lora_scale
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.lora_scale
                self.merged = True

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, ..., in_features)
        """

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # Base linear projection
        y_base = F.linear(x, T(self.weight), bias=self.bias)  # (B, ..., out_features)
        if self.pre_batch_norm:
            x = self.pre_bn(x)

        if self.r > 0 and not self.merged:
            p: torch.Tensor | None = ConditionRegistry.get(OP_PARAM_KEY)
            if p is not None and self._shared_p_projection is not None:
                p = p.to(self.lora_C.device)

                # Learnable representation of p: (B, p_dim)
                p_repr: torch.Tensor = self._shared_p_projection(p)

                # condition projection: (B, r) = (B, p_dim) @ (r, p_dim).T
                c = F.linear(p_repr, self.lora_C)

                # rank_space_shift: (B, r) = (B, p_dim) @ (r, p_dim).T
                s = F.linear(p_repr, self.lora_S)

                # input projection: (B, ..., r) = ( (B, ..., in_features) @ (r, in_features).T )
                h = F.linear(self.lora_dropout(x), self.lora_A)

                # introduce shift gate
                # only activate gate when x activates that rank direction
                # s_gate = torch.sigmoid(h.norm(dim=-1, keepdim=True))

                # Bilinear FiLM-inspired modulation in rank space
                # (B, ..., r)
                h_mod = h * (1.0 + expand_like(target=c, like=h)) + expand_like(
                    target=s, like=h
                )

                # Project back to output space
                # (B, ..., out_features) = ( (B, ..., r) @ (out_features, r).T )
                delta_y = F.linear(h_mod, self.lora_B) * self.lora_scale
                y_base += delta_y

            else:
                raise RuntimeError(
                    "Operating parameters not found or _shared_p_projection is None during forward pass."
                )

            if self.post_layer_norm:
                y_base = self.post_ln(y_base)

        return y_base


# Register shared projection
layer_dict = {
    "Linear": Linear,
    "VeRA": VeRALinear,
    "DoRA": DoRALinear,
    "LoRAM": LoRAMLinear,
    "BilinearLoRA": BilinearLoRA,
    "RSSBilinearLoRA": RSSBilinearLoRA,
}
