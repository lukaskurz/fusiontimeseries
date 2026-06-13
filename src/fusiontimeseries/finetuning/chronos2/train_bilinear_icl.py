"""In-context finetuning (ICF) of the Chronos-2 BilinearLoRA (Phase 9).

Phase 6 showed the finetuned model's in-context ability is INHERITED from base
pretraining — `train_bilinear.py` finetunes on single subsampled traces, never
on demonstrations. This entry point trains the SAME BilinearLoRA recipe on
multi-example ICL concatenations (`Chronos2ICLDataset`) at a larger context
window so the model learns to USE demonstrations, paired with level-clean
retrieval.

It reuses `train_bilinear`'s plumbing verbatim — `build_fts_config`,
`build_model`, `ConditionedTrainer`, `ensure_flat_flux_data`, the recipe
constants — and changes only:

- ``context_length`` raised to the ICF window (default 2048; a k=5 ICL stream
  of full ~267-step demos + query is ≤ 1522 steps). The OP-embedding dim
  ``op_embedding_dim=512`` is INDEPENDENT of the context window and is left
  untouched (`loralib/layers.py`); raising the window touches nothing in the
  conditioning path.
- ``batch_size`` reduced (default 32) so the wider window fits on MPS (batch
  128 @ 2048 OOMs).
- the dataset: ``Chronos2ICLDataset.train_val_split`` (train AND val build ICL
  concatenations; val retrieval/cutoff deterministic) instead of the
  single-trace ``Chronos2Dataset.train_val_split``.

Two checkpoints, one process each (``BilinearLoRA._shared_p_projection`` is a
CLASS attribute, so a second model load rebinds the shared embed under the
first — never train/eval two in one process):

- ``--icl-retrieval level``  → ``outputs/chronos2-bilinear-icl-level-<n>/``
- ``--icl-retrieval random`` → ``outputs/chronos2-bilinear-icl-random-<n>/`` (control)

Conditioning is QUERY-ONLY (Option A): the trainer patches one
``operating_parameters (B, 4)`` vector per forward = the query's params.

Usage:
    # smoke (~1 min): short run, frequent eval, confirm the loss path
    uv run python -m fusiontimeseries.finetuning.chronos2.train_bilinear_icl \
        --icl-retrieval level --max-steps 60 --eval-steps 20 \
        --batch-size 8 --device mps
    # full run (one per checkpoint, separate processes)
    uv run python -m fusiontimeseries.finetuning.chronos2.train_bilinear_icl \
        --icl-retrieval level  --max-steps 4000 --batch-size 32 --device mps
    uv run python -m fusiontimeseries.finetuning.chronos2.train_bilinear_icl \
        --icl-retrieval random --max-steps 4000 --batch-size 32 --device mps

Outputs mirror `train_bilinear.py`: ``lora_weights.pt`` (self-contained, loaded
by `few_shot/finetuned.py:load_finetuned_chronos2` with
``context_window=<ICF window>``), ``train_summary.json`` (+ the ICF metadata
block), ``trainable_params.json``, ``training_args.json``, ``fts_config.json``.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers.training_args import TrainingArguments

from fusiontimeseries.finetuning.chronos2.icl_dataset import Chronos2ICLDataset
from fusiontimeseries.finetuning.chronos2.train_bilinear import (
    DEFAULT_OUTPUT_BASE,
    ConditionedTrainer,
    build_fts_config,
    build_model,
    ensure_flat_flux_data,
)
from fusiontimeseries.lib.get_next_path import get_next_path
from fusiontimeseries.loralib.utils import (
    lora_state_dict,
    print_trainable_parameters,
)

#: ICF defaults (the user-locked design: window 2048, batch ~32, k ∈ {1,3,5}).
ICF_CONTEXT_LENGTH: int = 2048
ICF_BATCH_SIZE: int = 32
ICF_NUM_EXAMPLES: tuple[int, ...] = (1, 3, 5)
ICF_VAL_NUM_EXAMPLES: int = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos-2 BilinearLoRA in-context finetuning")
    parser.add_argument(
        "--icl-retrieval",
        choices=("level", "random"),
        required=True,
        help="level = demos retrieved by context-mean (train≡test ctx_level); "
        "random = control (demos sampled at random)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        nargs="+",
        default=list(ICF_NUM_EXAMPLES),
        help="k-set sampled per training sample",
    )
    parser.add_argument(
        "--val-num-examples",
        type=int,
        default=ICF_VAL_NUM_EXAMPLES,
        help="fixed k for the deterministic validation samples",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4000,
        help="60 (with small --eval-steps) = smoke, 4000 = the full recipe",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="eval/log/save cadence (recipe default 200; lower it for smoke)",
    )
    parser.add_argument("--context-length", type=int, default=ICF_CONTEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=ICF_BATCH_SIZE)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact output dir (default: outputs/chronos2-bilinear-icl-<mode>-<n>)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        DEFAULT_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        output_dir = get_next_path(
            base_fname=f"chronos2-bilinear-icl-{args.icl_retrieval}",
            base_dir=DEFAULT_OUTPUT_BASE,
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output directory created at: {output_dir}", flush=True)

    # Recipe config, then the ICF overrides (window + batch). build_fts_config
    # stays verbatim so the recipe knobs can never drift; we mutate only the
    # two fields the ICF design changes. op_embedding_dim (512) is unchanged.
    fts_config = build_fts_config(args.device, args.max_steps)
    fts_config.context_length = args.context_length
    fts_config.batch_size = args.batch_size
    fts_config.eval_steps = args.eval_steps
    fts_config.data_path = ensure_flat_flux_data()
    fts_config.save_config(output_dir / "fts_config.json")
    print(
        f"ICF config: retrieval={args.icl_retrieval}, k-set={tuple(args.num_examples)}, "
        f"context_length={fts_config.context_length}, batch_size={fts_config.batch_size}, "
        f"op_embedding_dim={fts_config.op_embedding_dim} (unchanged)",
        flush=True,
    )

    model = build_model(fts_config)
    print_trainable_parameters(model, save_path=output_dir / "trainable_params.json")

    train_dataset, val_dataset = Chronos2ICLDataset.train_val_split(
        fts_config,
        icl_retrieval=args.icl_retrieval,
        num_examples=tuple(args.num_examples),
        val_num_examples=args.val_num_examples,
    )

    on_cuda = args.device.startswith("cuda")
    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=fts_config.batch_size,
        per_device_eval_batch_size=fts_config.batch_size,
        learning_rate=fts_config.learning_rate,
        lr_scheduler_type=fts_config.lr_scheduler_type,
        warmup_ratio=fts_config.lr_scheduler_warmup_ratio,
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
        training_arguments._n_gpu = 1

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
    summary["icf"] = {
        "icl_retrieval": args.icl_retrieval,
        "num_examples": list(args.num_examples),
        "val_num_examples": args.val_num_examples,
        "context_length": fts_config.context_length,
        "batch_size": fts_config.batch_size,
    }
    with open(output_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    lora_weights = lora_state_dict(model)
    torch.save(lora_weights, output_dir / "lora_weights.pt")
    n_tensors = len(lora_weights)
    n_params = sum(v.numel() for v in lora_weights.values())
    print(
        f"Saved {output_dir / 'lora_weights.pt'} "
        f"({n_tensors} tensors, {n_params:,} params, "
        f"retrieval={args.icl_retrieval}, best eval_loss {trainer.state.best_metric}, "
        f"{train_seconds / 60:.1f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
