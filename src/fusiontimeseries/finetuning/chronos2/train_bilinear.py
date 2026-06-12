"""Script-ified Chronos-2 BilinearLoRA finetuning (chronos2_bilinear.ipynb cells 0-14).

Replicates Severin's recipe verbatim — fp32 ``amazon/chronos-2`` with
``chronos_config.context_length = 512``, BilinearLoRA (r=8, nominal alpha=16)
on q/k/v/o + output_patch_embedding.output_layer, a shared
``ContinuousConditionEmbed(512, 4)`` registered BOTH as the class-level
``BilinearLoRA._shared_p_projection`` AND as the ``shared_condition_projection``
submodule (so its ``lora_opc_embed`` MLP trains and lands in
``lora_weights.pt``), ``Chronos2Dataset.train_val_split`` over the
flux_data.json gyroswin train+val series, and the notebook's exact
FTSConfig/TrainingArguments — with three environment adaptations:

- ``optim="adamw_torch"`` on non-CUDA devices (FTSConfig freezes the literal
  ``adamw_torch_fused``, which is CUDA-oriented; the override goes straight to
  TrainingArguments, FTSConfig stays verbatim).
- no ``torch.cuda.empty_cache()`` / ``_n_gpu`` hack off CUDA.
- ``num_output_patches=ceil(prediction_length / output_patch_size)`` (= 5)
  passed in the trainer's forward: the installed chronos 2.2.x defaults it to
  1 and raises on the 80-step ``future_target``; the notebook ran an older
  version that inferred it.
- the tracked ``data/flux_data.json`` is the batch/gyroswin DICT dump, but
  ``lib/dataset.py:load_flux_data`` expects the older FLAT-LIST schema
  (FluxData records with idx/distribution) Severin's notebooks ran against.
  ``ensure_flat_flux_data()`` rebuilds that flat list from the gyroswin
  splits, recovering each entry's raw iteration id through the verified
  value-matched mapping (``few_shot/operating_params_mapping.json``) so
  ``is_train``/``is_validation``/``is_benchmark`` route exactly as upstream
  (gyroswin_id -> ID_BENCHMARK_IDXS, gyroswin_val -> ID_VALIDATION_IDXS,
  gyroswin_train -> TRAIN_IDXS; asserted at build time). The rebuilt file
  lands at ``data/flux/flux_data_flat.json`` (gitignored).

NOTE (recipe quirk, replicated on purpose): ``LoRALayer.convert`` drops
``lora_alpha`` in its recursion, so every converted layer gets the default
alpha=1 (lora_scale = 1/8), not the nominal 16. Severin's run had the same
behavior; reconstruction in ``few_shot/finetuned.py`` uses the identical
convert call so training and inference scales match.

Usage:
    uv run python -m fusiontimeseries.finetuning.chronos2.train_bilinear \
        --max-steps 200 --device mps     # pipeline test (~3 min)
    uv run python -m fusiontimeseries.finetuning.chronos2.train_bilinear \
        --max-steps 4000 --device mps    # full recipe (~1 h on M1 Max)

Outputs land in ``outputs/chronos2-bilinear-selftrained-<n>/`` (gitignored;
the recipe's save_steps checkpoints are ~480 MB full-model saves):
``lora_weights.pt`` (self-contained, ~7.5 MB), ``train_summary.json`` (incl.
log_history), ``trainable_params.json``, ``training_args.json``,
``fts_config.json``.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from chronos.chronos2.model import Chronos2Model, Chronos2Output
from transformers import Trainer
from transformers.training_args import TrainingArguments

from fusiontimeseries.finetuning.chronos2.dataset import Chronos2Dataset
from fusiontimeseries.lib.conditioning import ConditionRegistry
from fusiontimeseries.lib.config import FTSConfig
from fusiontimeseries.lib.get_next_path import get_next_path
from fusiontimeseries.lib.modules import ContinuousConditionEmbed
from fusiontimeseries.loralib.layers import BilinearLoRA
from fusiontimeseries.loralib.utils import (
    lora_state_dict,
    mark_only_lora_as_trainable,
    print_trainable_parameters,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_BASE: Path = REPO_ROOT / "outputs"
FLAT_FLUX_DATA_PATH: Path = REPO_ROOT / "data" / "flux" / "flux_data_flat.json"

# Notebook cell 6, verbatim.
TARGET_MODULE_NAMES: list[str] = [
    "self_attention.q",
    "self_attention.k",
    "self_attention.v",
    "self_attention.o",
    "output_patch_embedding.output_layer",
]
LORA_RANK: int = 8
LORA_ALPHA: int = 16


def ensure_flat_flux_data(out_path: Path = FLAT_FLUX_DATA_PATH) -> Path:
    """Rebuild the flat-list flux_data JSON that ``load_flux_data`` expects.

    The tracked ``data/flux_data.json`` is the batch/gyroswin dict dump;
    Severin's finetuning notebooks ran against an older flat list of FluxData
    records. This reconstructs that list from the gyroswin splits: each
    train/val/id entry gets its RAW iteration id back through the verified
    value-matched mapping (the dump keys 1000+i are permuted vs raw i), OOD
    entries get idx 0..4. The (energy_flux, params) pairing is intrinsic to
    each dump record — the mapping only restores the idx label that routes
    ``is_train`` / ``is_validation`` / ``is_benchmark``.

    Returns:
        Path to the rebuilt JSON (written only if missing).

    Raises:
        AssertionError: If the recovered raw ids do not land exactly on
            upstream's ID_BENCHMARK_IDXS / ID_VALIDATION_IDXS / TRAIN_IDXS.
    """
    if out_path.exists():
        return out_path

    from fusiontimeseries.benchmarking.few_shot.operating_params import load_mapping
    from fusiontimeseries.lib.config import FLUX_DATA_PATH
    from fusiontimeseries.lib.dataset import (
        ID_BENCHMARK_IDXS,
        ID_VALIDATION_IDXS,
        TRAIN_IDXS,
    )

    dump_key_to_raw_id: dict[tuple[str, str], int] = {
        (entry["dump_split"], entry["dump_key"]): int(raw_id)
        for raw_id, entry in load_mapping()["raw_traces"].items()
        if entry["dump_key"] is not None
    }
    dump: dict = json.load(open(FLUX_DATA_PATH, "r"))

    flat: list[dict] = []
    recovered: dict[str, list[int]] = {}
    for split in ("gyroswin_train", "gyroswin_val", "gyroswin_id", "gyroswin_ood"):
        recovered[split] = []
        for key, entry in dump[split].items():
            if split == "gyroswin_ood":
                idx, distribution = int(key) - 4000, "ood"
            else:
                idx, distribution = dump_key_to_raw_id[(split, key)], "id"
            recovered[split].append(idx)
            flat.append(
                {
                    "idx": idx,
                    "distribution": distribution,
                    "shat": entry["shat"],
                    "q": entry["q"],
                    "rlt": entry["rlt"],
                    "rln": entry["rln"],
                    "energy_flux": entry["energy_flux"],
                }
            )

    assert sorted(recovered["gyroswin_id"]) == sorted(ID_BENCHMARK_IDXS), (
        f"gyroswin_id raw ids {sorted(recovered['gyroswin_id'])} != ID_BENCHMARK_IDXS"
    )
    assert sorted(recovered["gyroswin_val"]) == sorted(ID_VALIDATION_IDXS), (
        f"gyroswin_val raw ids {sorted(recovered['gyroswin_val'])} != ID_VALIDATION_IDXS"
    )
    outside = [i for i in recovered["gyroswin_train"] if i not in TRAIN_IDXS]
    assert not outside, f"gyroswin_train raw ids outside TRAIN_IDXS: {outside}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(flat, f)
    print(f"Rebuilt flat flux_data ({len(flat)} entries) -> {out_path}", flush=True)
    return out_path


def build_fts_config(device: str, max_steps: int) -> FTSConfig:
    """Notebook cell 2, verbatim (device/max_steps are the CLI knobs)."""
    return FTSConfig(
        op_embedding_dim=512,
        num_ops=4,
        context_length=512,
        pred_tail_timestamps=80,
        batch_size=128,
        stratification_bins=5,
        sampling_bins=5,
        val_size=0.1,
        padding_value=torch.nan,
        padding_mask_default=0.0,
        padding_mask_indicator=1.0,
        stratification="opc_pca",
        sampling_strategy="linear",
        data_augmentation="white_noise",
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        lr_scheduler_warmup_ratio=0.0,
        optimizer_type="adamw_torch_fused",
        max_grad_norm=1.0,
        max_steps=max_steps,
        eval_steps=200,
        gradient_accumulation_steps=1,
        device=device,
    )


def build_model(fts_config: FTSConfig) -> Chronos2Model:
    """Notebook cells 3-8: base model + BilinearLoRA conversion + embed wiring."""
    # LoRA-init reproducibility (kaiming inits in convert + the condition
    # embed); the HF Trainer seeds everything else via TrainingArguments
    # seed=42 and the dataset seeds numpy with FTSConfig.random_seed=123.
    torch.manual_seed(42)

    model = Chronos2Model.from_pretrained("amazon/chronos-2")
    model.chronos_config.context_length = fts_config.context_length
    model = model.to(fts_config.device)

    shared_p_projection = ContinuousConditionEmbed(
        embedding_dim=fts_config.op_embedding_dim,
        n_cond=fts_config.num_ops,
        max_wavelength=10_000,
        init_weights="kaiming_uniform",
    )
    BilinearLoRA._shared_p_projection = shared_p_projection

    model = BilinearLoRA.convert(
        module=model,
        kind="BilinearLoRA",
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_module_names=TARGET_MODULE_NAMES,
    )
    # Register the shared projection as a submodule so it gets saved/loaded
    # properly (its MLP is named lora_opc_embed -> trainable + in the
    # lora_state_dict).
    model.shared_condition_projection = shared_p_projection
    model.shared_condition_projection.to(fts_config.device)

    mark_only_lora_as_trainable(model=model, bias="none")
    return model


class ConditionedTrainer(Trainer):
    """Notebook cell 11 + num_output_patches for the installed chronos 2.2.x."""

    num_output_patches: int = 1

    def compute_loss(
        self,
        model: Chronos2Model,
        inputs: dict[str, torch.Tensor],
        *args,
        return_outputs=False,
        **kwargs,
    ):
        # Tensor[B, N]
        p_raw: torch.Tensor | None = inputs.pop(
            "operating_parameters", None
        )  # remove before forward, otherwise TypeError in Trainer
        assert p_raw is not None, "operating_parameters key is missing in inputs"

        with ConditionRegistry.patch(op_params=p_raw):
            outputs: Chronos2Output = model(
                **inputs, num_output_patches=self.num_output_patches
            )

        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos-2 BilinearLoRA finetuning")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4000,
        help="200 = pipeline test, 4000 = the full notebook recipe",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output dir (default: outputs/chronos2-bilinear-selftrained-<n>)",
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if args.output_dir is None:
        DEFAULT_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        output_dir = get_next_path(
            base_fname="chronos2-bilinear-selftrained", base_dir=DEFAULT_OUTPUT_BASE
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output directory created at: {output_dir}", flush=True)

    fts_config = build_fts_config(args.device, args.max_steps)
    fts_config.data_path = ensure_flat_flux_data()
    fts_config.save_config(output_dir / "fts_config.json")

    model = build_model(fts_config)
    print_trainable_parameters(model, save_path=output_dir / "trainable_params.json")

    train_dataset, val_dataset = Chronos2Dataset.train_val_split(fts_config)

    on_cuda = args.device.startswith("cuda")
    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=fts_config.batch_size,
        per_device_eval_batch_size=fts_config.batch_size,
        learning_rate=fts_config.learning_rate,
        lr_scheduler_type=fts_config.lr_scheduler_type,
        warmup_ratio=fts_config.lr_scheduler_warmup_ratio,
        # FTSConfig freezes the CUDA-oriented fused literal; MPS/CPU need the
        # plain torch AdamW (same optimizer maths, no fused kernel).
        optim=fts_config.optimizer_type if on_cuda else "adamw_torch",
        logging_strategy="steps",
        logging_steps=fts_config.eval_steps,
        disable_tqdm=False,
        report_to="none",
        max_steps=fts_config.max_steps,
        gradient_accumulation_steps=fts_config.gradient_accumulation_steps,
        dataloader_num_workers=0,
        tf32=False,
        bf16=False,
        save_only_model=True,
        prediction_loss_only=True,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=fts_config.eval_steps,
        eval_strategy="steps",
        eval_steps=fts_config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        use_cpu=False,
        label_names=["future_target"],
        remove_unused_columns=False,
        max_grad_norm=fts_config.max_grad_norm,
    )
    if on_cuda:
        training_arguments._n_gpu = 1  # notebook hack, pointless off CUDA

    trainer = ConditionedTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    output_patch_size = getattr(model.chronos_config, "output_patch_size", 16)
    trainer.num_output_patches = math.ceil(
        fts_config.prediction_length / output_patch_size
    )
    print(
        f"num_output_patches = {trainer.num_output_patches} "
        f"(prediction_length {fts_config.prediction_length} / "
        f"output_patch_size {output_patch_size})",
        flush=True,
    )
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(trainer.args.to_dict(), f, indent=4)

    t0 = time.perf_counter()
    train_output = trainer.train()
    train_seconds = time.perf_counter() - t0

    summary = dict(train_output._asdict())
    summary["train_seconds"] = train_seconds
    summary["log_history"] = trainer.state.log_history
    summary["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    summary["best_metric"] = trainer.state.best_metric
    with open(output_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    lora_weights = lora_state_dict(model)
    torch.save(lora_weights, output_dir / "lora_weights.pt")
    n_tensors = len(lora_weights)
    n_params = sum(v.numel() for v in lora_weights.values())
    print(
        f"Saved {output_dir / 'lora_weights.pt'} "
        f"({n_tensors} tensors, {n_params:,} params, "
        f"best eval_loss {trainer.state.best_metric}, "
        f"{train_seconds / 60:.1f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
