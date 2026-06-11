"""Model-free baselines for the few-shot benchmark (Phase 1).

All baselines return a full-length series: the ground-truth 80-step context
followed by a constant level — the benchmark metric only reads the mean of
the last 80 timesteps, so a constant level is the natural model-free
prediction shape.

- persistence: level = last context value.
- pool tail-mean: level = mean over the example pool's tail means (the
  "global saturation level" prior).
- kNN-copy: level = mean of the k nearest pool traces' tail means, with
  nearest measured by Euclidean distance between z-scored 80-step contexts.
  Retrieval without any TSFM — bounds what retrieval alone achieves and
  competes with the GyroSwin paper's GPR baseline (43.8 ID).
"""

import numpy as np
from numpy.typing import NDArray

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    FewShotExample,
)
from fusiontimeseries.benchmarking.few_shot.harness import ForecastFn

__all__ = [
    "persistence_forecast",
    "make_pool_tail_mean_forecast",
    "make_knn_copy_forecast",
]


def _constant_after_context(
    trace: NDArray[np.float32], level: float, context_length: int
) -> NDArray[np.float32]:
    """Ground-truth context followed by a constant level, aligned with trace."""
    context = trace[:context_length]
    tail = np.full(len(trace) - len(context), level, dtype=trace.dtype)
    return np.concatenate([context, tail])


def persistence_forecast(
    trace: NDArray[np.float32],
    examples: list[FewShotExample],
    config: FewShotConfig,
) -> NDArray[np.float32]:
    """Persistence baseline: repeat the last context value."""
    context = trace[: config.start_context_length]
    return _constant_after_context(trace, float(context[-1]), config.start_context_length)


def make_pool_tail_mean_forecast(
    pool: list[FewShotExample], tail: int = 80
) -> ForecastFn:
    """Baseline predicting the example pool's global mean saturation level.

    Args:
        pool: Example pool (FIXED pool, test ids excluded).
        tail: Tail window (in subsampled steps) for per-trace tail means.
    """
    level = float(np.mean([np.mean(ex.trace_array[-tail:]) for ex in pool]))

    def forecast_fn(
        trace: NDArray[np.float32],
        examples: list[FewShotExample],
        config: FewShotConfig,
    ) -> NDArray[np.float32]:
        return _constant_after_context(trace, level, config.start_context_length)

    return forecast_fn


def make_knn_copy_forecast(
    pool: list[FewShotExample],
    k: int = 1,
    tail: int = 80,
    rescale: bool = False,
) -> ForecastFn:
    """kNN-copy baseline: retrieve nearest pool traces, copy their tail level.

    Distance is Euclidean between z-scored contexts (query first 80 steps vs
    each pool example's 80-step context). Ties are broken by pool order
    (stable sort).

    Args:
        pool: Example pool (FIXED pool, test ids excluded).
        k: Number of neighbours to average.
        tail: Tail window for per-trace tail means.
        rescale: If False, the level is the mean of the neighbours' RAW tail
            means. If True, each neighbour's tail mean is z-scored with its
            own context scaler and mapped back through the QUERY context
            scaler (amplitude transfer instead of absolute copy).
    """
    contexts_z: list[NDArray[np.float64]] = []
    ctx_stats: list[tuple[float, float]] = []
    tail_means: list[float] = []
    for ex in pool:
        ctx = ex.context_array.astype(np.float64)
        mean, std = float(ctx.mean()), float(ctx.std())
        std = std if std > 0 else 1.0
        contexts_z.append((ctx - mean) / std)
        ctx_stats.append((mean, std))
        tail_means.append(float(np.mean(ex.trace_array[-tail:])))
    pool_matrix = np.stack(contexts_z)

    def forecast_fn(
        trace: NDArray[np.float32],
        examples: list[FewShotExample],
        config: FewShotConfig,
    ) -> NDArray[np.float32]:
        query = trace[: config.start_context_length].astype(np.float64)
        if len(query) != pool_matrix.shape[1]:
            raise ValueError(
                f"Query context length {len(query)} != pool context length "
                f"{pool_matrix.shape[1]}"
            )
        q_mean, q_std = float(query.mean()), float(query.std())
        q_std = q_std if q_std > 0 else 1.0
        query_z = (query - q_mean) / q_std

        distances = np.linalg.norm(pool_matrix - query_z, axis=1)
        neighbours = np.argsort(distances, kind="stable")[:k]

        if rescale:
            levels = [
                q_mean + ((tail_means[i] - ctx_stats[i][0]) / ctx_stats[i][1]) * q_std
                for i in neighbours
            ]
        else:
            levels = [tail_means[i] for i in neighbours]
        return _constant_after_context(
            trace, float(np.mean(levels)), config.start_context_length
        )

    return forecast_fn


if __name__ == "__main__":
    from fusiontimeseries.benchmarking.few_shot.few_shot_utils import create_example_pool
    from fusiontimeseries.benchmarking.few_shot.harness import (
        DEFAULT_SAVE_DIR,
        load_results,
        paired_comparison,
        results_table,
        run_benchmark,
    )
    from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS

    print("Running model-free baselines...")

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    assert not ({ex.trace_id for ex in pool} & ID_TEST_RAW_IDS), "Test-set leakage in pool!"
    assert len(pool) == 245, f"Expected fixed pool of 245, got {len(pool)}"

    config = FewShotConfig(
        device="cpu",
        model_slug="baseline",
        model_prediction_length=64,
        start_context_length=80,
        relevant_prediction_tail=80,
        k_shot=0,
        random_seed=42,
        example_target_length=None,
    )

    baselines: dict[str, ForecastFn] = {
        "persistence": persistence_forecast,
        "pool_tail_mean": make_pool_tail_mean_forecast(pool),
        "knn_copy_k1": make_knn_copy_forecast(pool, k=1),
        "knn_copy_k5": make_knn_copy_forecast(pool, k=5),
    }

    runs = {}
    for method, forecast_fn in baselines.items():
        runs[method] = run_benchmark(
            forecast_fn=forecast_fn,
            config=config,
            example_pool=pool,
            method=method,
            deterministic=True,
        )

    print("\n--- Baseline results ---")
    table = results_table([run.model_dump() for run in runs.values()])
    print(
        table[["method", "k", "id_rmse", "id_se", "ood_rmse", "ood_se"]].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"
        )
    )
    print("\nReference points: GyroSwin-paper GPR 43.8 ID; TiRex k=5 few-shot ~30-42 ID")

    print("\n--- Paired comparison: persistence vs knn_copy_k1 ---")
    for split in ("in_distribution", "out_of_distribution"):
        comparison = paired_comparison(runs["persistence"], runs["knn_copy_k1"], split)
        print(
            f"  {split}: diff={comparison.rmse_diff:+.2f} "
            f"(persistence {comparison.rmse_a:.2f} vs knn_copy_k1 {comparison.rmse_b:.2f}), "
            f"wilcoxon_p={comparison.wilcoxon_p}, "
            f"bootstrap CI=[{comparison.bootstrap_ci_low:.2f}, {comparison.bootstrap_ci_high:.2f}], "
            f"p={comparison.bootstrap_p:.3f}"
        )

    print(f"\n✅ Baselines done — results in {DEFAULT_SAVE_DIR}")
