from pathlib import Path
from typing import Literal

import json

from pydantic import BaseModel, Field
import torch

__all__ = ["FTSConfig"]

T_OP = Literal["shat", "q", "rlt", "rln"]
OP_NAMES: list[T_OP] = ["shat", "q", "rlt", "rln"]
FLUX_DATA_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "flux_data.json"
)
assert FLUX_DATA_PATH.exists() and FLUX_DATA_PATH.is_file(), (
    f"No file at {FLUX_DATA_PATH}"
)


class FTSConfig(BaseModel):
    """Fusion Time Series Config"""

    op_embedding_dim: int = 512
    num_ops: int = 4
    context_length: int = 512
    start_context_timestamp: int = 40  # the timestamp at which the context starts; this is used to determine the range of timestamps to sample from for the context
    train_context_cutoffs: list[int] = Field(
        default_factory=lambda: [1, 70, 139]
    )  # the cutoff timestamps to use for training; these are the timestamps at which the context ends and the target starts; this is used to determine the range of timestamps to sample from for the target during training
    prediction_length: int = 80
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    op_ranges: dict[T_OP, tuple[float, float]] = Field(
        default_factory=lambda: {
            "shat": (0, 5),
            "q": (1, 9),
            "rlt": (3.5, 12),
            "rln": (0, 7),
        }
    )
    random_seed: int = 123
    stratification_bins: int = 5
    sampling_bins: int = 5
    val_size: float = 0.1
    padding_value: float = 0.0  # value to use for padding in context and target
    padding_mask_default: float = 0.0  # default value of padding mask
    padding_mask_indicator: float = (
        1.0  # this value is present in the mask tensor if the position is padded
    )
    pred_tail_timestamps: Literal[80] = 80
    subsample_factor: Literal[3] = 3
    full_subsampling: bool = True  # subsample all in-between windows
    stratification: Literal["tail_mean", "opc_pca"] | None = None
    sampling_strategy: Literal["linear", "pca_cluster", "tail_mean"] = "linear"
    data_augmentation: Literal["white_noise", "random_walk"] | None = None
    drop_ids: list[int] = Field(default_factory=list)
    data_path: Path = FLUX_DATA_PATH

    learning_rate: float = 1e-4
    lr_scheduler_type: Literal["linear", "constant"] = "linear"
    lr_scheduler_warmup_ratio: float = 0.0
    optimizer_type: Literal["adamw_torch_fused"] = "adamw_torch_fused"
    max_grad_norm: float = 1.0  # default in Trainer as well.

    max_steps: int = 5_000
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500

    def save_config(self, path: Path) -> None:
        """Save the config to a file.

        Args:
            path (Path): The path to save the config to.
        """

        with open(path, "w") as f:
            json.dump(self.model_dump(exclude={"data_path", "dtype"}), f, indent=4)

    class Config:
        """Pydantic config for FTSConfig"""

        extra = "forbid"
        arbitrary_types_allowed = True
