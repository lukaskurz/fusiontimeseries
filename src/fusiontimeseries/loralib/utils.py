#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from typing import Dict

from .layers import LoRALayer, DoRALinear, LoRAMLinear


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True  # type: ignore
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def init_scaling(m: nn.Module):
    if isinstance(m, DoRALinear):
        m.set_weight_norm()
    elif isinstance(m, LoRAMLinear):
        m.register_weight_parametrization()


def drop_ranks(exp_var_dict: dict, state_dict: dict, r: int, threshold: float = 0.9):
    ct_dict = {}
    remaining_exp_var = {}
    for key in exp_var_dict.keys():
        if key.startswith("roberta"):
            exp_var_cumsum = np.cumsum(exp_var_dict[key])
            mask = exp_var_cumsum < threshold
            inds_smaller_thresh = sum(mask)
            if not inds_smaller_thresh:
                inds_smaller_thresh = 1
            to_ind = r if r < inds_smaller_thresh else inds_smaller_thresh
            if to_ind == r:
                remaining_exp_var[key] = (
                    exp_var_dict[key][to_ind:],
                    state_dict[key][to_ind:, :],
                )
            state_dict[key] = state_dict[key][:to_ind, :]
            state_dict[key.replace("lora_A", "lora_B")] = state_dict[
                key.replace("lora_A", "lora_B")
            ][:, :to_ind]
            ct_dict[key] = to_ind
    return state_dict, remaining_exp_var, ct_dict


def redistribute_ranks(
    state_dict: dict,
    exp_vars: dict,
    threshold: float = 0.9,
    rank: int = 0,
    from_scratch: bool = False,
):
    # create new state dict according to redistribution of ranks
    ct = len([state_dict[k] for k in state_dict.keys() if "lora_A" in k])
    rank_budget = rank * ct
    if not from_scratch:
        new_state_dict, remaining_exp_vars, layer_count_dict = drop_ranks(
            exp_vars, state_dict, rank, threshold
        )
    else:
        layer_count_dict = {k: 0 for k in state_dict.keys()}
        new_state_dict = {k: v for k, v in state_dict.items() if "lora" not in k}
        remaining_exp_vars = {
            k: (exp_var, state_dict[k])
            for k, exp_var in exp_vars.items()
            if "lora_A" in k
        }
    n_ranks = sum(layer_count_dict.values())
    importance_list = [
        (k, v, c)
        for k, (value, components) in remaining_exp_vars.items()
        for v, c in zip(value, components)
        if "classifier" not in k
    ]
    importance_list.sort(key=lambda x: x[1])
    while n_ranks < rank_budget:
        # redistribute ranks according to explained variances
        key, _, comp = importance_list.pop()
        if key not in new_state_dict:
            new_state_dict[key] = comp.reshape(1, -1)
        else:
            new_state_dict[key] = torch.cat([new_state_dict[key], comp.reshape(1, -1)])

        if new_state_dict.get(key.replace("lora_A", "lora_B"), None) is not None:
            # delete lora_B keys, since they are zeros anyways
            del new_state_dict[key.replace("lora_A", "lora_B")]

        n_ranks += 1
    return new_state_dict


def print_trainable_parameters(model: nn.Module, save_path: Path | None = None) -> None:
    """Prints and optionally saves trainable parameters information for a PyTorch model.
    This function iterates through all parameters in the model, counting total and
    trainable parameters. It prints the number of trainable parameters for each layer
    and provides a summary of total trainable vs. total parameters. Optionally saves
    the detailed information to a JSON file.
    Args:
        model (nn.Module): The PyTorch model to analyze.
        save_path (Path | None, optional): Path to save the parameter information as JSON.
            If None, the information is only printed. Defaults to None.
    Returns:
        None
    Examples:
        >>> model = MyModel()
        >>> print_trainable_parameters(model)
        layer1.weight: 1,024
        layer2.weight: 2,048
        Trainable parameters: 3,072 / 10,000 (30.72%)
        >>> print_trainable_parameters(model, save_path=Path("params.json"))
        # Prints parameter info and saves to params.json
    """

    trainable_params = 0
    total_params = 0
    param_info = {}
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            param_info[name] = param.numel()
            # print(f"{name}: {param.numel():,}")

    summary = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": 100 * trainable_params / total_params,
    }
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({summary['trainable_percentage']:.2f}%)"
    )

    if save_path is not None:
        result = {"parameters": param_info, "summary": summary}
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)
