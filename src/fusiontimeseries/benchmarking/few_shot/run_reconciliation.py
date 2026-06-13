"""Phase-8 reconciliation run: the adaptation ladder on Severin's ``[0::3]`` phase.

Re-runs the ladder cells on ``Phase0BenchmarkProvider`` (267-step ``[0::3]``
traces, the phase Severin's finetuning eval uses) under our protocol (shared
scaling, mean decoding, honest ``[-80:]`` tail metric — the harness default) so a
single, internally-consistent ladder exists across both halves of the project.
Single seed 42 throughout (all cells are deterministic: k=0, deterministic
retrieval, or model-free baselines).

Cells (10) -> ``results/few_shot_v8_reconciliation/``:

- base Chronos-2 (bf16):       zeroshot k=0, mmr_euclid k=5
- Chronos-Bolt-tiny (bf16):    zeroshot k=0, mmr_euclid k=10
- finetuned BilinearLoRA:      zeroshot k=0, mmr_euclid k=5 (full 8192 window)
- finetuned @ 512 window:      mmr_euclid k=5
- model-free baselines:        persistence, kNN-copy k=5, pool tail-mean

``oracle_tail`` is excluded: ``selection`` ranks it against true tail means
cached for the ``[2::3]`` provider (re-keying is out of scope; every ladder rung
here is phase-correct).

Result JSONs carry a ``protocol``/``trace_phase`` marker and a ``*-phase0`` method
token, and use the ``*_reconciliation.json`` suffix — so they stay invisible to
``load_results`` (``*_fewshot_results.json``) and never mix with ``[2::3]`` runs.
The finetuned model is loaded ONE at a time (``BilinearLoRA._shared_p_projection``
is a class attribute): base -> bolt -> ft-full -> ft-512, baselines on CPU.

Usage (smoke first):
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_reconciliation \
        --checkpoint outputs/chronos2-bilinear-selftrained-0/lora_weights.pt \
        --smoke --device mps
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_reconciliation \
        --checkpoint outputs/chronos2-bilinear-selftrained-0/lora_weights.pt \
        --device mps
"""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from fusiontimeseries.benchmarking.few_shot.baselines import (
    make_knn_copy_forecast,
    make_pool_tail_mean_forecast,
    persistence_forecast,
)
from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    create_example_pool,
)
from fusiontimeseries.benchmarking.few_shot.finetuned import (
    FINETUNED_SLUG,
    FT_TRAIN_CONTEXT,
    checkpoint_id,
    load_finetuned_chronos2,
)
from fusiontimeseries.benchmarking.few_shot.harness import (
    ForecastFn,
    FewShotRunResults,
    run_benchmark,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
from fusiontimeseries.benchmarking.few_shot.presentation import (
    make_concat_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.reconciliation import (
    PHASE0_STEPS,
    Phase0BenchmarkProvider,
    make_phase0_finetuned_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    MODEL_SLUGS,
    chronos2_predict_from_pipeline,
    make_chronos2_pipeline,
    make_chronos_bolt_predict,
)
from fusiontimeseries.benchmarking.few_shot.selection import make_select_fn

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v8_reconciliation"

POINT_STAT: str = "mean"  # Phase-5 default for Chronos-2 / Bolt / finetuned


class ReconciliationRunner:
    """Shared plumbing: Phase-0 provider, fixed pool, save dir, mmr selector."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.device: str = args.device
        self.save_dir: Path = args.save_dir
        self.provider = Phase0BenchmarkProvider(device=args.device)
        self.pool = create_example_pool(
            exclude_ids=set(ID_TEST_RAW_IDS), target_length=None
        )
        assert len(self.pool) == 245, f"Expected fixed pool of 245, got {len(self.pool)}"
        leaked = sorted({ex.trace_id for ex in self.pool} & ID_TEST_RAW_IDS)
        assert not leaked, f"Leakage guard: ID test raw ids in pool: {leaked}"
        self.mmr = make_select_fn("mmr_euclid")

    def make_config(
        self,
        slug: str,
        k: int,
        checkpoint: str | None = None,
        model_context_window: int | None = None,
    ) -> FewShotConfig:
        return FewShotConfig(
            device=self.device,
            model_slug=slug,
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=k,
            random_seed=42,
            example_target_length=None,
            normalization="shared",
            point_stat=POINT_STAT,
            checkpoint=checkpoint,
            model_context_window=model_context_window,
        )

    def run_cell(
        self,
        forecast_fn: ForecastFn,
        slug: str,
        strategy: str,
        k: int,
        method: str,
        select_fn=None,
        checkpoint: str | None = None,
        model_context_window: int | None = None,
    ) -> None:
        config = self.make_config(slug, k, checkpoint, model_context_window)
        t0 = time.perf_counter()
        results = run_benchmark(
            forecast_fn=forecast_fn,
            config=config,
            example_pool=self.pool,
            method=method,
            select_fn=select_fn,
            seeds=(42,),
            deterministic=True,
            provider=self.provider,
            save=False,
        )
        out_path = save_reconciliation(results, self.save_dir, method, k)
        print(
            f"[{method}] k={k}: ID {results.in_distribution.rmse:.2f}, "
            f"OOD {results.out_of_distribution.rmse:.2f} "
            f"[{time.perf_counter() - t0:.0f}s] -> {out_path.name}",
            flush=True,
        )

    def cleanup(self, *objects) -> None:
        del objects
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


def save_reconciliation(
    results: FewShotRunResults, save_dir: Path, method: str, k: int
) -> Path:
    """Write a result JSON tagged as a Phase-0 reconciliation cell.

    Adds ``protocol``/``trace_phase``/``trace_steps`` markers and uses the
    ``*_reconciliation.json`` suffix so the file is distinguishable from a
    ``[2::3]`` run and stays off the ``load_results`` glob.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    data = results.model_dump()
    data["protocol"] = "phase0_reconciliation"
    data["trace_phase"] = "[0::3]"
    data["trace_steps"] = PHASE0_STEPS
    out_path = save_dir / f"{results.timestamp}_{method}_k{k}_reconciliation.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def method_label(slug: str, strategy: str, win: int | None = None) -> str:
    """``{slug}_{strategy}__shared-mean[-win512]-phase0`` for a model cell."""
    tokens = ["shared", "mean"]
    if win is not None:
        tokens.append(f"win{win}")
    tokens.append("phase0")
    return f"{slug.replace('/', '_')}_{strategy}__{'-'.join(tokens)}"


########################################################
# Cells
########################################################


def run_base_chronos2(runner: ReconciliationRunner) -> None:
    slug = MODEL_SLUGS["chronos2"]
    print(f"\n=== BASE {slug} (bf16) on {runner.device} ===", flush=True)
    pipeline = make_chronos2_pipeline(runner.device)
    forecast_fn = make_concat_forecast_fn(
        chronos2_predict_from_pipeline(pipeline, POINT_STAT), "shared"
    )
    runner.run_cell(forecast_fn, slug, "zeroshot", 0, method_label(slug, "zeroshot"))
    runner.run_cell(
        forecast_fn, slug, "mmr_euclid", 5, method_label(slug, "mmr_euclid"),
        select_fn=runner.mmr,
    )
    runner.cleanup(pipeline, forecast_fn)


def run_bolt(runner: ReconciliationRunner) -> None:
    slug = MODEL_SLUGS["chronos_bolt"]
    print(f"\n=== {slug} (bf16) on {runner.device} ===", flush=True)
    predict_fn = make_chronos_bolt_predict(runner.device, POINT_STAT)
    forecast_fn = make_concat_forecast_fn(predict_fn, "shared")
    runner.run_cell(forecast_fn, slug, "zeroshot", 0, method_label(slug, "zeroshot"))
    runner.run_cell(
        forecast_fn, slug, "mmr_euclid", 10, method_label(slug, "mmr_euclid"),
        select_fn=runner.mmr,
    )
    runner.cleanup(predict_fn, forecast_fn)


def run_finetuned_full(runner: ReconciliationRunner, checkpoint: Path) -> None:
    ck = checkpoint_id(checkpoint)
    print(
        f"\n=== FINETUNED {FINETUNED_SLUG} (fp32, full window) on {runner.device} "
        f"[{ck}] ===",
        flush=True,
    )
    pipeline = load_finetuned_chronos2(checkpoint, runner.device)
    forecast_fn = make_phase0_finetuned_forecast_fn(
        pipeline, runner.provider, POINT_STAT, "shared"
    )
    runner.run_cell(
        forecast_fn, FINETUNED_SLUG, "zeroshot", 0,
        method_label(FINETUNED_SLUG, "zeroshot"), checkpoint=ck,
    )
    runner.run_cell(
        forecast_fn, FINETUNED_SLUG, "mmr_euclid", 5,
        method_label(FINETUNED_SLUG, "mmr_euclid"), select_fn=runner.mmr,
        checkpoint=ck,
    )
    runner.cleanup(pipeline, forecast_fn)


def run_finetuned_win512(runner: ReconciliationRunner, checkpoint: Path) -> None:
    ck = checkpoint_id(checkpoint)
    print(
        f"\n=== FINETUNED {FINETUNED_SLUG} (fp32, window {FT_TRAIN_CONTEXT}) on "
        f"{runner.device} [{ck}] ===",
        flush=True,
    )
    pipeline = load_finetuned_chronos2(
        checkpoint, runner.device, context_window=FT_TRAIN_CONTEXT
    )
    forecast_fn = make_phase0_finetuned_forecast_fn(
        pipeline, runner.provider, POINT_STAT, "shared"
    )
    runner.run_cell(
        forecast_fn, FINETUNED_SLUG, "mmr_euclid", 5,
        method_label(FINETUNED_SLUG, "mmr_euclid", win=FT_TRAIN_CONTEXT),
        select_fn=runner.mmr, checkpoint=ck, model_context_window=FT_TRAIN_CONTEXT,
    )
    runner.cleanup(pipeline, forecast_fn)


def run_baselines(runner: ReconciliationRunner) -> None:
    print(f"\n=== model-free baselines (CPU) ===", flush=True)
    baselines: dict[str, ForecastFn] = {
        "persistence": persistence_forecast,
        "pool_tail_mean": make_pool_tail_mean_forecast(runner.pool),
        "knn_copy_k5": make_knn_copy_forecast(runner.pool, k=5),
    }
    for method, forecast_fn in baselines.items():
        runner.run_cell(forecast_fn, "baseline", method, 0, f"{method}-phase0")


########################################################
# Smoke
########################################################


def smoke(runner: ReconciliationRunner, args: argparse.Namespace) -> None:
    print(f"\n=== Smoke: reconciliation checks on {args.device} ===", flush=True)

    # S1 — provider sanity: 11 keys, 267 steps
    n = len(list(runner.provider.items()))
    assert n == 11, f"S1: expected 11 Phase-0 traces, got {n}"
    sample = runner.provider.get_id("iteration_8_ifft").numpy()
    assert sample.shape == (PHASE0_STEPS,) and np.all(np.isfinite(sample))
    print(f"✓ S1: Phase0 provider — 11 traces, {PHASE0_STEPS} steps, finite")

    # S2 — leakage guard (already asserted in __init__, restated explicitly)
    leaked = sorted({ex.trace_id for ex in runner.pool} & ID_TEST_RAW_IDS)
    assert not leaked, f"S2: ID test raw ids leaked into pool: {leaked}"
    print(f"✓ S2: leakage guard — none of {sorted(ID_TEST_RAW_IDS)} in the 245-trace pool")

    # S3 — one real base zeroshot trace through the [0::3] provider, finite
    slug = MODEL_SLUGS["chronos2"]
    pipeline = make_chronos2_pipeline(args.device)
    forecast_fn = make_concat_forecast_fn(
        chronos2_predict_from_pipeline(pipeline, POINT_STAT), "shared"
    )
    config = runner.make_config(slug, 0)
    trace = runner.provider.get_id("iteration_8_ifft").numpy()
    t0 = time.perf_counter()
    out = forecast_fn(trace, [], config)
    dt = time.perf_counter() - t0
    assert out.shape == trace.shape and np.all(np.isfinite(out)), (
        "S3: non-finite base zeroshot forecast on the [0::3] trace"
    )
    assert np.array_equal(out[:80], trace[:80]), "S3: context not passed through"
    print(
        f"✓ S3: base zeroshot on the [0::3] iteration_8 trace finite "
        f"(tail {float(np.mean(out[-80:])):.2f}, true {float(np.mean(trace[-80:])):.2f}, "
        f"{dt:.1f}s)"
    )
    runner.cleanup(pipeline, forecast_fn)
    print("\n✅ Smoke test passed (no results written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-8 evaluation reconciliation run")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="lora_weights.pt for the finetuned cells",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Provider sanity + one real base zeroshot trace + leakage guard; exit",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    assert args.checkpoint.exists(), f"No checkpoint at {args.checkpoint}"

    runner = ReconciliationRunner(args)
    if args.smoke:
        smoke(runner, args)
        return

    t0 = time.perf_counter()
    run_base_chronos2(runner)
    run_bolt(runner)
    run_finetuned_full(runner, args.checkpoint)
    run_finetuned_win512(runner, args.checkpoint)
    run_baselines(runner)
    print(
        f"\n✅ Reconciliation run complete [{(time.perf_counter() - t0) / 60:.0f} min] "
        f"-> {args.save_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
