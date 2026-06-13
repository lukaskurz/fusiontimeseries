"""Phase-9 eval grid: ICL on the IN-CONTEXT-finetuned Chronos-2 (ICF).

Phase 6 found the finetuned model's in-context ability is INHERITED from base
pretraining (it was finetuned on single traces). Phase 9 trains the BilinearLoRA
ON demonstrations (`finetuning/chronos2/train_bilinear_icl.py` →
`Chronos2ICLDataset`) and asks two questions here:

1. **Did it learn to USE demonstrations?** — compare the ``level``-ICF checkpoint
   (demos retrieved by context level, train≡test) against the ``random``-control
   checkpoint (demos sampled at random during training). Separation ⇒ the model
   learned to exploit level-matched demos rather than ignore/tolerate them.
2. **Does demonstration-training beat the inherited ICL ability?** — compare the
   ICF rungs against the v6 single-trace finetuned model.

Each checkpoint is loaded at its 2048 TRAINING window
(`load_finetuned_chronos2(..., context_window=2048)`) and evaluated through the
FROZEN Phase-3 shared-scaling rollout (`make_finetuned_forecast_fn(..,
"shared")`) — protocol-identical to the v6 grid except for the window. The two
checkpoints get distinct slugs (`amazon/chronos-2-bilinear-icf-{level,random}`)
so their cells never collide; every cell records the checkpoint sha256.

``BilinearLoRA._shared_p_projection`` is a CLASS attribute → ONE finetuned model
per process. Run the level and random checkpoints in SEPARATE invocations.

Cells per checkpoint × {median, mean}:
- ``zeroshot`` k=0
- ``ctx_level`` k5/k10, ``mmr_euclid`` k5/k10, ``mmr_level`` k5/k10
- ``oracle_tail`` k10 (cheating ceiling)
- ``random`` k10 (multi-seed control: do random demos erase the ICF advantage?)

Usage (smoke first, then one process per checkpoint):
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_icl_finetuned \
        --checkpoint outputs/chronos2-bilinear-icl-level-0/lora_weights.pt \
        --icl-retrieval level --smoke --device mps
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_icl_finetuned \
        --checkpoint outputs/chronos2-bilinear-icl-level-0/lora_weights.pt \
        --icl-retrieval level --device mps
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_icl_finetuned \
        --checkpoint outputs/chronos2-bilinear-icl-random-0/lora_weights.pt \
        --icl-retrieval random --device mps
"""

import argparse
import time
from pathlib import Path

import numpy as np

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import FewShotConfig
from fusiontimeseries.benchmarking.few_shot.finetuned import (
    checkpoint_id,
    load_finetuned_chronos2,
    make_finetuned_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.harness import ForecastFn
from fusiontimeseries.benchmarking.few_shot.run_presentation_grid import (
    CellRunner,
    seeds_for,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v9_icl"

#: Distinct slug per ICF checkpoint so level/random cells never collide and the
#: analyzer can separate them. (The v6 single-trace ft slug is different again.)
ICF_SLUGS: dict[str, str] = {
    "level": "amazon/chronos-2-bilinear-icf-level",
    "random": "amazon/chronos-2-bilinear-icf-random",
}

#: ICF training/eval window (the dataset's context_length; see train_bilinear_icl).
ICF_EVAL_WINDOW: int = 2048

POINT_STATS: tuple[str, ...] = ("median", "mean")
ICL_CONFIGS: tuple[tuple[str, int], ...] = (
    ("ctx_level", 5),
    ("ctx_level", 10),
    ("mmr_euclid", 5),
    ("mmr_euclid", 10),
    ("mmr_level", 5),
    ("mmr_level", 10),
    ("oracle_tail", 10),
)
RANDOM_K: int = 10


def run_checkpoint(runner: CellRunner, args: argparse.Namespace) -> None:
    slug = ICF_SLUGS[args.icl_retrieval]
    ck = checkpoint_id(args.checkpoint)
    print(
        f"\n=== ICF {slug} (fp32, window {ICF_EVAL_WINDOW}) on {args.device} [{ck}] ===",
        flush=True,
    )
    pipeline = load_finetuned_chronos2(
        args.checkpoint, args.device, context_window=ICF_EVAL_WINDOW
    )
    for point_stat in POINT_STATS:
        forecast_fn: ForecastFn = make_finetuned_forecast_fn(pipeline, point_stat, "shared")
        runner.run_cell(
            forecast_fn, slug, "zeroshot", 0, None, (42,), True,
            normalization="shared", point_stat=point_stat, checkpoint=ck,
            model_context_window=ICF_EVAL_WINDOW,
        )
        for strategy, k in ICL_CONFIGS:
            seeds, deterministic = seeds_for(strategy, args.random_seeds)
            runner.run_cell(
                forecast_fn, slug, strategy, k, runner.select_fns[strategy],
                seeds, deterministic,
                normalization="shared", point_stat=point_stat, checkpoint=ck,
                model_context_window=ICF_EVAL_WINDOW,
            )
        runner.run_cell(
            forecast_fn, slug, "random", RANDOM_K, runner.select_fns["random"],
            tuple(range(args.random_seeds)), False,
            normalization="shared", point_stat=point_stat, checkpoint=ck,
            model_context_window=ICF_EVAL_WINDOW,
        )
        runner.cleanup(forecast_fn)
    runner.cleanup(pipeline)


def smoke(runner: CellRunner, args: argparse.Namespace) -> None:
    print(f"\n=== Smoke: ICF eval checks on {args.device} ===", flush=True)
    ck = checkpoint_id(args.checkpoint)
    pipeline = load_finetuned_chronos2(
        args.checkpoint, args.device, context_window=ICF_EVAL_WINDOW
    )
    assert pipeline.model.chronos_config.context_length == ICF_EVAL_WINDOW
    print(f"✓ S1: {ck} loads at window {ICF_EVAL_WINDOW} (exact LoRA key set + shapes)")

    provider = runner.provider
    trace = provider.get_id("iteration_8_ifft").numpy()
    slug = ICF_SLUGS[args.icl_retrieval]
    for point_stat in ("median", "mean"):
        forecast_fn = make_finetuned_forecast_fn(pipeline, point_stat, "shared")
        examples = runner.select_fns["ctx_level"](
            runner.pool, 5, 42, trace[:80], "iteration_8_ifft"
        )
        config = FewShotConfig(
            device=args.device, model_slug=slug, model_prediction_length=64,
            start_context_length=80, relevant_prediction_tail=80, k_shot=5,
            random_seed=42, example_target_length=None, normalization="shared",
            point_stat=point_stat, checkpoint=ck, model_context_window=ICF_EVAL_WINDOW,
        )
        t0 = time.perf_counter()
        out = forecast_fn(trace, examples, config)
        dt = time.perf_counter() - t0
        assert out.shape == trace.shape and np.all(np.isfinite(out)), "non-finite forecast"
        assert np.array_equal(out[:80], trace[:80]), "context not passed through"
        print(
            f"✓ S2 [{point_stat}]: ctx_level k=5 forecast finite, context "
            f"passthrough, tail mean {np.mean(out[-80:]):.2f} [{dt:.1f}s -> "
            f"det cell ≈ {11 * dt:.0f}s]"
        )
        runner.cleanup(forecast_fn)
    runner.cleanup(pipeline)
    print("\n✅ Smoke test passed (no results written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-9 ICF eval grid")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--icl-retrieval", choices=("level", "random"), required=True,
        help="which ICF checkpoint this is (sets the result slug)",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    assert args.checkpoint.exists(), f"No checkpoint at {args.checkpoint}"

    runner = CellRunner(args)
    if args.smoke:
        smoke(runner, args)
        return

    t0 = time.perf_counter()
    run_checkpoint(runner, args)
    print(
        f"\n✅ ICF eval ({args.icl_retrieval}) complete "
        f"[{(time.perf_counter() - t0) / 60:.0f} min]",
        flush=True,
    )


if __name__ == "__main__":
    main()
