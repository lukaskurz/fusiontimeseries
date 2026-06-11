"""Phase-2 selection-strategy grid: {strategy} x {k} x {model}.

Runs every example-selection strategy through the Phase-1 harness with the
FIXED 245-trace pool and the t266 protocol (full-length example targets,
context 80, prediction 64, tail 80). Deterministic strategies run a single
pass (seed 42); ``random`` runs ``--random-seeds`` seeds for a fair mean+-std
band. One k=0 zero-shot anchor per model unless ``--skip-k0``.

Method labels are ``{model_slug_with_underscores}_{strategy}`` (e.g.
``NX-AI_TiRex_op_knn``) so files stay parseable by ``load_results`` and the
strategy can be recovered by stripping the model prefix.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_selection_grid \
        --device mps --include-mmr

Resumable: re-invoke with ``--models``/``--strategies``/``--ks`` subsets; the
analysis globs whatever is in the save dir.
"""

import argparse
import gc
from pathlib import Path

import torch

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    create_example_pool,
)
from fusiontimeseries.benchmarking.few_shot.harness import (
    make_icl_forecast_fn,
    run_benchmark,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    MODEL_SLUGS,
    PREDICT_FACTORIES,
)
from fusiontimeseries.benchmarking.few_shot.selection import (
    STRATEGIES,
    make_select_fn,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import BenchmarkDataProvider

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v2_selection"

#: Strategies run by default; mmr_euclid is opt-in via --include-mmr.
DEFAULT_STRATEGIES: tuple[str, ...] = (
    "random",
    "op_knn",
    "ctx_euclid",
    "ctx_dtw",
    "ctx_growth",
    "oracle_tail",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 selection-strategy grid")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SLUGS),
        default=["tirex", "timesfm", "chronos2", "chronos_bolt"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=None,
        help="Strategy subset (default: all but mmr_euclid; see --include-mmr)",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=20,
        help="Number of seeds (0..n-1) for the random strategy",
    )
    parser.add_argument(
        "--skip-k0", action="store_true", help="Skip the per-model k=0 zero-shot anchor"
    )
    parser.add_argument(
        "--include-mmr",
        action="store_true",
        help="Add mmr_euclid to the default strategy set",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()

    if args.strategies is not None:
        strategies = tuple(args.strategies)
    else:
        strategies = DEFAULT_STRATEGIES + (("mmr_euclid",) if args.include_mmr else ())

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    assert len(pool) == 245, f"Expected fixed pool of 245, got {len(pool)}"
    provider = BenchmarkDataProvider()

    # Built ONCE so the per-query ranking caches (DTW!) are shared across models.
    select_fns = {strategy: make_select_fn(strategy) for strategy in strategies}

    def make_config(slug: str, k: int) -> FewShotConfig:
        return FewShotConfig(
            device=args.device,
            model_slug=slug,
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=k,
            random_seed=42,
            example_target_length=None,
        )

    for model_name in args.models:
        slug = MODEL_SLUGS[model_name]
        model_clean = slug.replace("/", "_")
        print(f"\n{'=' * 60}\nLoading {slug} on {args.device}\n{'=' * 60}", flush=True)
        predict_fn = PREDICT_FACTORIES[model_name](args.device)
        forecast_fn = make_icl_forecast_fn(predict_fn)

        if not args.skip_k0:
            results = run_benchmark(
                forecast_fn=forecast_fn,
                config=make_config(slug, 0),
                example_pool=pool,
                method=f"{model_clean}_zeroshot",
                seeds=(42,),
                deterministic=True,
                provider=provider,
                save_dir=args.save_dir,
            )
            print(
                f"[{model_name}] zeroshot k=0: "
                f"ID {results.in_distribution.rmse:.2f}, "
                f"OOD {results.out_of_distribution.rmse:.2f}",
                flush=True,
            )

        for strategy in strategies:
            for k in args.ks:
                if strategy == "random":
                    seeds = tuple(range(args.random_seeds))
                    deterministic = False
                else:
                    seeds = (42,)
                    deterministic = True
                results = run_benchmark(
                    forecast_fn=forecast_fn,
                    config=make_config(slug, k),
                    example_pool=pool,
                    method=f"{model_clean}_{strategy}",
                    select_fn=select_fns[strategy],
                    seeds=seeds,
                    deterministic=deterministic,
                    provider=provider,
                    save_dir=args.save_dir,
                )
                std = results.in_distribution.rmse_std_seeds
                std_note = f" (std {std:.2f})" if std is not None else ""
                print(
                    f"[{model_name}] {strategy} k={k}: "
                    f"ID {results.in_distribution.rmse:.2f}{std_note}, "
                    f"OOD {results.out_of_distribution.rmse:.2f}",
                    flush=True,
                )

        del predict_fn, forecast_fn
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    print("\n✅ Selection grid complete", flush=True)


if __name__ == "__main__":
    main()
