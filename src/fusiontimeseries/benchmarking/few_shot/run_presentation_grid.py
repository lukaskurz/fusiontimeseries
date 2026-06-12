"""Phase-3 presentation-format grid: ablations x strategies x k x models.

Four staged ablations, all writing to ONE save dir
(``results/few_shot_v3_presentation/``) so every headline comparison lives
within a single grid run (MPS is not bit-deterministic across runs):

- ``group``  (Stage A, chronos2 only): Chronos-2 group ICL vs flat concat on
  identical example sets — ks {1,3,5,10,20} x {random, ctx_euclid,
  oracle_tail} x {__base, __group} + both k=0 anchors.
- ``norm``   (Stage B, all models): shared scaling (__shared) full grid (all
  7 strategies) + fresh __base anchors (random multi-seed; ctx_euclid +
  oracle_tail deterministic; k=0) per model.
- ``order``  (Stage C, all models): ctx_euclid example ordering —
  __simfirst (deterministic) and __shuforder (multi-seed) vs the Stage-B
  __base anchor (similar_last). k=1 is order-free, hence ks {3,5,10}.
- ``trunc``  (Stage D, tirex + timesfm): post-selection truncation
  (peak+64) under ``--trunc-norm`` x {random, ctx_euclid} x ks {5,10,20} +
  full-example comparison cells at k=20 (k<=10 full cells come from Stage B).

Method labels are ``{slug with / -> _}_{strategy}__{variant}`` where
``variant`` is the hyphen-joined non-default tokens (``base`` | ``group``;
``shared``; ``simfirst`` | ``shuforder``; ``trunc64``), e.g.
``amazon_chronos-2_random__group`` or
``google_timesfm-2.5-200m-pytorch_random__shared-trunc64``.

A cell whose result file already exists in the save dir is skipped — this
both makes re-invocations resumable and dedupes the chronos2 __base anchors
shared between Stages A and B.

Usage (staged background jobs; D after B's norm verdict):
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_presentation_grid \
        --ablations group --device mps
    ... --ablations norm
    ... --ablations order
    ... --ablations trunc --trunc-norm shared

``--smoke`` loads the real Chronos-2 and runs three sanity checks (constant
series through group ICL, group-k0 vs concat-k0, one k=2 timing pass)
without writing results.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    create_example_pool,
)
from fusiontimeseries.benchmarking.few_shot.harness import (
    ForecastFn,
    SelectFn,
    run_benchmark,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
from fusiontimeseries.benchmarking.few_shot.presentation import (
    make_chronos2_group_forecast_fn,
    make_concat_forecast_fn,
    make_ordered_select_fn,
    make_truncated_select_fn,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    MODEL_SLUGS,
    PREDICT_FACTORIES,
    chronos2_predict_from_pipeline,
    make_chronos2_pipeline,
)
from fusiontimeseries.benchmarking.few_shot.selection import (
    STRATEGIES,
    make_select_fn,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import BenchmarkDataProvider

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v3_presentation"

ABLATIONS: tuple[str, ...] = ("group", "norm", "order", "trunc")
STAGE_KS: dict[str, list[int]] = {
    "group": [1, 3, 5, 10, 20],
    "norm": [1, 3, 5, 10],
    "order": [3, 5, 10],
    "trunc": [5, 10, 20],
}
TRUNC_MARGIN: int = 64
GROUP_STRATEGIES: tuple[str, ...] = ("random", "ctx_euclid", "oracle_tail")
BASE_ANCHOR_STRATEGIES: tuple[str, ...] = ("random", "ctx_euclid", "oracle_tail")
TRUNC_STRATEGIES: tuple[str, ...] = ("random", "ctx_euclid")
TRUNC_MODELS: tuple[str, ...] = ("tirex", "timesfm")


def variant_label(
    presentation: str = "concat",
    normalization: str = "per_example",
    example_order: str = "similar_last",
    trunc_margin: int | None = None,
    op_covariates: str | None = None,
) -> str:
    """Hyphen-joined non-default tokens; 'base' if everything is default."""
    tokens: list[str] = []
    if presentation == "group":
        tokens.append("group")
    if normalization == "shared":
        tokens.append("shared")
    if example_order == "similar_first":
        tokens.append("simfirst")
    elif example_order == "shuffled":
        tokens.append("shuforder")
    if trunc_margin is not None:
        tokens.append(f"trunc{trunc_margin}")
    if op_covariates == "step":
        tokens.append("opcov")
    elif op_covariates == "permuted":
        tokens.append("permcov")
    return "-".join(tokens) if tokens else "base"


def cell_exists(save_dir: Path, method: str, k: int) -> bool:
    return any(save_dir.glob(f"*_{method}_k{k}_fewshot_results.json"))


def seeds_for(strategy: str, n_random_seeds: int) -> tuple[tuple[int, ...], bool]:
    """(seeds, deterministic) — random gets the seed axis, the rest seed 42."""
    if strategy == "random":
        return tuple(range(n_random_seeds)), False
    return (42,), True


class CellRunner:
    """Shared plumbing for one grid invocation (pool/provider/save dir)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.device: str = args.device
        self.save_dir: Path = args.save_dir
        self.pool = create_example_pool(
            exclude_ids=set(ID_TEST_RAW_IDS), target_length=None
        )
        assert len(self.pool) == 245, f"Expected fixed pool of 245, got {len(self.pool)}"
        self.provider = BenchmarkDataProvider()
        # Built ONCE so per-query ranking caches are shared across all stages.
        self.select_fns: dict[str, SelectFn] = {
            strategy: make_select_fn(strategy) for strategy in STRATEGIES
        }

    def run_cell(
        self,
        forecast_fn: ForecastFn,
        slug: str,
        strategy: str,
        k: int,
        select_fn: SelectFn | None,
        seeds: tuple[int, ...],
        deterministic: bool,
        presentation: str = "concat",
        normalization: str = "per_example",
        example_order: str = "similar_last",
        trunc_margin: int | None = None,
        op_covariates: str | None = None,
    ) -> None:
        variant = variant_label(
            presentation, normalization, example_order, trunc_margin, op_covariates
        )
        method = f"{slug.replace('/', '_')}_{strategy}__{variant}"
        if cell_exists(self.save_dir, method, k):
            print(f"[skip] {method} k={k}: already in {self.save_dir.name}", flush=True)
            return
        config = FewShotConfig(
            device=self.device,
            model_slug=slug,
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=k,
            random_seed=seeds[0],
            example_target_length=None,
            presentation=presentation,
            normalization=normalization,
            example_order=example_order,
            example_truncation_margin=trunc_margin,
            op_covariates=op_covariates,
        )
        t0 = time.perf_counter()
        results = run_benchmark(
            forecast_fn=forecast_fn,
            config=config,
            example_pool=self.pool,
            method=method,
            select_fn=select_fn,
            seeds=seeds,
            deterministic=deterministic,
            provider=self.provider,
            save_dir=self.save_dir,
        )
        std = results.in_distribution.rmse_std_seeds
        std_note = f" (std {std:.2f})" if std is not None else ""
        print(
            f"[{method}] k={k}: ID {results.in_distribution.rmse:.2f}{std_note}, "
            f"OOD {results.out_of_distribution.rmse:.2f} "
            f"[{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )

    def cleanup(self, *objects) -> None:
        del objects
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


########################################################
# Stages
########################################################


def stage_group(runner: CellRunner, args: argparse.Namespace) -> None:
    """Stage A: Chronos-2 group ICL vs flat concat on identical example sets."""
    slug = MODEL_SLUGS["chronos2"]
    ks = args.ks or STAGE_KS["group"]
    print(f"\n=== Stage A (group): {slug} on {args.device}, ks={ks} ===", flush=True)
    pipeline = make_chronos2_pipeline(args.device)
    concat_fn = make_concat_forecast_fn(chronos2_predict_from_pipeline(pipeline))
    group_fn = make_chronos2_group_forecast_fn(pipeline)
    presentations: tuple[tuple[str, ForecastFn], ...] = (
        ("concat", concat_fn),
        ("group", group_fn),
    )
    for presentation, forecast_fn in presentations:
        runner.run_cell(
            forecast_fn, slug, "zeroshot", 0, None, (42,), True, presentation=presentation
        )
    for strategy in GROUP_STRATEGIES:
        seeds, deterministic = seeds_for(strategy, args.random_seeds)
        for k in ks:
            # Same select_fn + same seeds => identical example sets across
            # presentations (the analyzer hard-asserts this).
            for presentation, forecast_fn in presentations:
                runner.run_cell(
                    forecast_fn,
                    slug,
                    strategy,
                    k,
                    runner.select_fns[strategy],
                    seeds,
                    deterministic,
                    presentation=presentation,
                )
    runner.cleanup(pipeline, concat_fn, group_fn)


def stage_norm(runner: CellRunner, args: argparse.Namespace) -> None:
    """Stage B: shared scaling full grid + fresh __base anchors per model."""
    ks = args.ks or STAGE_KS["norm"]
    anchor_seeds = args.base_anchor_seeds or args.random_seeds
    for model_name in args.models:
        slug = MODEL_SLUGS[model_name]
        print(f"\n=== Stage B (norm): {slug} on {args.device}, ks={ks} ===", flush=True)
        predict_fn = PREDICT_FACTORIES[model_name](args.device)
        base_fn = make_concat_forecast_fn(predict_fn, "per_example")
        shared_fn = make_concat_forecast_fn(predict_fn, "shared")

        # k=0 anchor (shared == per_example at k=0 by construction)
        runner.run_cell(base_fn, slug, "zeroshot", 0, None, (42,), True)

        # Fresh __base anchors within this grid run (chronos2 dedupes vs Stage A)
        for strategy in BASE_ANCHOR_STRATEGIES:
            seeds, deterministic = seeds_for(strategy, args.random_seeds)
            if strategy == "random":
                seeds = tuple(range(anchor_seeds))
            for k in ks:
                runner.run_cell(
                    base_fn, slug, strategy, k, runner.select_fns[strategy], seeds, deterministic
                )

        # __shared full grid: all 7 strategies
        for strategy in STRATEGIES:
            seeds, deterministic = seeds_for(strategy, args.random_seeds)
            for k in ks:
                runner.run_cell(
                    shared_fn,
                    slug,
                    strategy,
                    k,
                    runner.select_fns[strategy],
                    seeds,
                    deterministic,
                    normalization="shared",
                )
        runner.cleanup(predict_fn, base_fn, shared_fn)


def stage_order(runner: CellRunner, args: argparse.Namespace) -> None:
    """Stage C: ctx_euclid ordering ablation (anchor = Stage-B __base)."""
    ks = args.ks or STAGE_KS["order"]
    for model_name in args.models:
        slug = MODEL_SLUGS[model_name]
        print(f"\n=== Stage C (order): {slug} on {args.device}, ks={ks} ===", flush=True)
        predict_fn = PREDICT_FACTORIES[model_name](args.device)
        base_fn = make_concat_forecast_fn(predict_fn, "per_example")
        simfirst_select = make_ordered_select_fn(runner.select_fns["ctx_euclid"], "similar_first")
        shuffled_select = make_ordered_select_fn(runner.select_fns["ctx_euclid"], "shuffled")
        for k in ks:
            runner.run_cell(
                base_fn, slug, "ctx_euclid", k, simfirst_select, (42,), True,
                example_order="similar_first",
            )
            # Shuffled ordering varies with the seed even though ctx_euclid
            # selection does not — run a real (small) seed axis.
            runner.run_cell(
                base_fn, slug, "ctx_euclid", k, shuffled_select,
                tuple(range(args.order_seeds)), False,
                example_order="shuffled",
            )
        runner.cleanup(predict_fn, base_fn)


def stage_trunc(runner: CellRunner, args: argparse.Namespace) -> None:
    """Stage D: post-selection truncation (peak+64) under --trunc-norm."""
    ks = args.ks or STAGE_KS["trunc"]
    normalization = args.trunc_norm
    models = [m for m in args.models if m in TRUNC_MODELS]
    for model_name in models:
        slug = MODEL_SLUGS[model_name]
        print(
            f"\n=== Stage D (trunc): {slug} on {args.device}, ks={ks}, "
            f"norm={normalization} ===",
            flush=True,
        )
        predict_fn = PREDICT_FACTORIES[model_name](args.device)
        forecast_fn = make_concat_forecast_fn(predict_fn, normalization)
        for strategy in TRUNC_STRATEGIES:
            trunc_select = make_truncated_select_fn(
                runner.select_fns[strategy], margin=TRUNC_MARGIN
            )
            seeds, deterministic = seeds_for(strategy, args.random_seeds)
            for k in ks:
                runner.run_cell(
                    forecast_fn, slug, strategy, k, trunc_select, seeds, deterministic,
                    normalization=normalization,
                    trunc_margin=TRUNC_MARGIN,
                )
        # Full-example comparison cells at the largest k (smaller-k full cells
        # come from Stage B under the same normalization).
        k_max = max(ks)
        if model_name == "tirex" and args.skip_tirex_full_kmax:
            print(f"[skip] {slug} full-example k={k_max} cells (--skip-tirex-full-kmax)")
        else:
            for strategy in TRUNC_STRATEGIES:
                seeds, deterministic = seeds_for(strategy, args.random_seeds)
                runner.run_cell(
                    forecast_fn, slug, strategy, k_max, runner.select_fns[strategy],
                    seeds, deterministic,
                    normalization=normalization,
                )
        runner.cleanup(predict_fn, forecast_fn)


########################################################
# Smoke test (real Chronos-2, no results written)
########################################################


def smoke(runner: CellRunner, args: argparse.Namespace) -> None:
    print(f"\n=== Smoke test: real Chronos-2 group ICL on {args.device} ===", flush=True)
    pipeline = make_chronos2_pipeline(args.device)
    concat_fn = make_concat_forecast_fn(chronos2_predict_from_pipeline(pipeline))
    group_fn = make_chronos2_group_forecast_fn(pipeline)

    def make_config(k: int) -> FewShotConfig:
        return FewShotConfig(
            device=args.device,
            model_slug=MODEL_SLUGS["chronos2"],
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=k,
            random_seed=42,
            example_target_length=None,
        )

    examples = runner.select_fns["ctx_euclid"](
        runner.pool, 2, 42,
        runner.provider.get_id("iteration_8_ifft").numpy()[:80],
        "iteration_8_ifft",
    )

    # 1. Constant series through group ICL (NaN-padded dict task on device)
    rng = np.random.default_rng(0)
    const_trace = (5.0 + 0.01 * rng.normal(size=266)).astype(np.float32)
    out = group_fn(const_trace, examples, make_config(2))
    assert np.all(np.isfinite(out)), "Smoke: non-finite group forecast on constant series"
    print(
        f"✓ constant series, group k=2: finite; tail mean "
        f"{np.mean(out[-80:]):.3f} (true 5.0)"
    )

    # 2. group k=0 vs concat k=0 (same model, two input paths)
    trace = runner.provider.get_id("iteration_8_ifft").numpy()
    g0 = group_fn(trace, [], make_config(0))
    c0 = concat_fn(trace, [], make_config(0))
    rel = float(np.linalg.norm(g0 - c0) / np.linalg.norm(c0))
    assert np.all(np.isfinite(g0)), "Smoke: non-finite group k=0 forecast"
    assert rel < 0.05, f"Smoke: group k=0 deviates from concat k=0 (rel {rel:.4f})"
    print(f"✓ group k=0 ≈ concat k=0 (relative L2 diff {rel:.5f})")

    # 3. Timing calibration: one full-trace group pass at k=2
    t0 = time.perf_counter()
    out2 = group_fn(trace, examples, make_config(2))
    dt = time.perf_counter() - t0
    assert np.all(np.isfinite(out2)), "Smoke: non-finite group k=2 forecast"
    print(
        f"✓ group k=2 single trace: {dt:.1f}s -> est. {11 * dt:.0f}s per "
        f"11-trace pass"
    )
    print("\n✅ Smoke test passed (no results written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3 presentation-format grid")
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=ABLATIONS,
        default=list(ABLATIONS),
        help="Which stages to run (A=group, B=norm, C=order, D=trunc)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SLUGS),
        default=["tirex", "timesfm", "chronos2", "chronos_bolt"],
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=None,
        help="Override the per-stage default ks (applies to all selected stages)",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=20,
        help="Number of seeds (0..n-1) for the random strategy",
    )
    parser.add_argument(
        "--base-anchor-seeds",
        type=int,
        default=None,
        help="Budget fallback: fewer seeds for Stage B's __base random anchors",
    )
    parser.add_argument(
        "--order-seeds",
        type=int,
        default=5,
        help="Number of seeds for the shuffled-ordering cells (Stage C)",
    )
    parser.add_argument(
        "--trunc-norm",
        choices=("per_example", "shared"),
        default="shared",
        help="Normalization for Stage D (decide after Stage B's verdict)",
    )
    parser.add_argument(
        "--skip-tirex-full-kmax",
        action="store_true",
        help="Budget fallback: skip TiRex's full-example k=20 comparison cells",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the real-Chronos-2 sanity checks and exit (writes nothing)",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()

    runner = CellRunner(args)
    if args.smoke:
        smoke(runner, args)
        return

    stages = {"group": stage_group, "norm": stage_norm, "order": stage_order, "trunc": stage_trunc}
    for ablation in args.ablations:
        stages[ablation](runner, args)
    print("\n✅ Presentation grid stage(s) complete", flush=True)


if __name__ == "__main__":
    main()
