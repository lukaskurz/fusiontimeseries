import json
from pathlib import Path

from pydantic import BaseModel
from transformers.training_args import TrainingArguments

from fusiontimeseries.lib.get_next_path import get_next_path


def get_output_dir(base_dir: str, base_folder: str) -> Path:
    _base_dir = Path(base_dir)
    _base_dir.mkdir(parents=True, exist_ok=True)
    output_dir = get_next_path(base_fname=base_folder, base_dir=_base_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output directory created at: {output_dir}")
    return output_dir


class FinetuningConfig(BaseModel):
    """Finetuning Config"""

    context_length: int = 288
    eval_context_cutoff: int
    train_context_cutoffs: list[int]
    prediction_length: int = 128
    batch_size: int = 128
    padding_value: float = 0.0  # value to use for padding in context and target
    padding_mask_default: float = 0.0  # default value of padding mask
    padding_mask_indicator: float = (
        1.0  # this value is present in the mask tensor if the position is padded
    )
    subsampling: bool = True
    random_seed: int = 42

    learning_rate: float = 1e-3
    max_steps: int = 3_000
    eval_steps: int = 200

    def save_config(self, path: Path) -> None:
        """Save the config to a file.

        Args:
            path (Path): The path to save the config to.
        """

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    class Config:
        """Pydantic config for FTSConfig"""

        extra = "forbid"
        arbitrary_types_allowed = True

    def get_training_arguments(
        self, output_dir: Path, load_best_model_at_end: bool
    ) -> TrainingArguments:
        training_arguments = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            lr_scheduler_type="linear",
            optim="adamw_torch_fused",
            logging_strategy="steps",
            logging_steps=self.eval_steps,
            disable_tqdm=False,
            report_to="none",
            max_steps=self.max_steps,
            gradient_accumulation_steps=1,
            dataloader_num_workers=0,
            tf32=False,
            bf16=False,
            save_only_model=True,
            save_total_limit=2,
            save_strategy="steps",
            save_steps=self.eval_steps,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            load_best_model_at_end=load_best_model_at_end,
            dataloader_drop_last=False,
            greater_is_better=False,
            use_cpu=False,
            label_names=[
                "future_target"
            ],  # must be truthy for HF Trainer to use overridden compute_loss
            remove_unused_columns=False,  # needed to not accidentally remove columns that our custom compute_loss relies on
            max_grad_norm=1.0,
        )
        training_arguments._n_gpu = 1
        return training_arguments
