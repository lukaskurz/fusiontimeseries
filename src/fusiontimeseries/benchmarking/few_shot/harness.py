"""Evaluation harness for few-shot ICL benchmarks (Phase 1).

Provides the pieces every later phase reports through:
- ``make_icl_forecast_fn``: the notebooks' autoregressive ICL rollout,
  extracted verbatim and parameterized over a model-specific ``PredictFn``.
- ``run_benchmark``: multi-seed evaluation over the 6 ID + 5 OOD benchmark
  traces with per-trace records, persisted as a backward-compatible superset
  of the existing results-JSON schema.
- ``paired_comparison``: Wilcoxon signed-rank + paired bootstrap between two
  runs over per-trace squared errors.
- ``load_results`` / ``results_table``: aggregation across result files
  (including legacy single-seed files from results/few_shot_t{80,266}).
"""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pydantic import BaseModel
from scipy import stats
from sklearn.preprocessing import StandardScaler

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    FewShotExample,
    select_examples_random,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    IN_DISTRIBUTION_ITERATIONS,
    OUT_OF_DISTRIBUTION_ITERATIONS,
    BenchmarkDataProvider,
    rmse_with_standard_error,
)

__all__ = [
    "DEFAULT_SEEDS",
    "DEFAULT_SAVE_DIR",
    "PredictFn",
    "ForecastFn",
    "SelectFn",
    "make_icl_forecast_fn",
    "run_benchmark",
    "paired_comparison",
    "load_results",
    "results_table",
    "TraceResult",
    "SeedResult",
    "SplitSummary",
    "FewShotRunResults",
    "PairedComparison",
]

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v2"
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(20))

LEGACY_FILENAME_RE = re.compile(r"^(\d{8}_\d{6})_(?P<method>.+)_k(?P<k>\d+)_fewshot_results$")


class PredictFn(Protocol):
    """Model-specific single-step prediction in normalized space.

    Model wrappers (TiRex / Chronos-2 / Chronos-Bolt / TimesFM) live in the
    notebooks or runner scripts; the harness only sees this signature.
    """

    def __call__(
        self, context: NDArray[np.float32], prediction_length: int
    ) -> NDArray[np.float32]:
        """Forecast ``prediction_length`` steps from a 1D context.

        Args:
            context: 1D (already normalized) ICL context.
            prediction_length: Number of steps to predict.

        Returns:
            1D median/point forecast of length ``prediction_length``.
        """
        ...


class ForecastFn(Protocol):
    """Full-trace forecast: ground-truth context + autoregressive rollout."""

    def __call__(
        self,
        trace: NDArray[np.float32],
        examples: list[FewShotExample],
        config: FewShotConfig,
    ) -> NDArray[np.float32]:
        """Return a full-length series aligned with ``trace``."""
        ...


class SelectFn(Protocol):
    """Example-selection strategy.

    ``query_context`` makes the signature ready for Phase-2 retrieval-based
    selection; random selection ignores it.
    """

    def __call__(
        self,
        pool: list[FewShotExample],
        k: int,
        seed: int,
        query_context: NDArray[np.float32],
    ) -> list[FewShotExample]: ...


def _select_random(
    pool: list[FewShotExample],
    k: int,
    seed: int,
    query_context: NDArray[np.float32],
) -> list[FewShotExample]:
    """Default SelectFn: seeded random selection (ignores the query)."""
    return select_examples_random(pool, k=k, seed=seed)


def make_icl_forecast_fn(predict_fn: PredictFn) -> ForecastFn:
    """Wrap a model's PredictFn into the standard autoregressive ICL rollout.

    This is the notebooks' ``fewshot_autoregressive_forecast`` extracted
    verbatim: per-example StandardScaler fit on the example context, query
    scaler fit on the first ``start_context_length`` steps, then a loop that
    concatenates ``[ex_ctx, ex_tgt] * k + query`` in normalized space,
    predicts ``model_prediction_length`` steps, denormalizes/appends, and
    re-normalizes the extended query until the trace length is covered.

    ``examples=[]`` reduces to the plain zero-shot protocol (k=0).
    """

    def forecast_fn(
        trace: NDArray[np.float32],
        examples: list[FewShotExample],
        config: FewShotConfig,
    ) -> NDArray[np.float32]:
        trace_length = trace.shape[0]

        # Normalize examples independently
        normalized_examples = []
        for ex in examples:
            ex_scaler = StandardScaler()
            normed_ctx = ex_scaler.fit_transform(ex.context_array.reshape(-1, 1)).squeeze()
            normed_tgt = ex_scaler.transform(ex.target_array.reshape(-1, 1)).squeeze()
            normalized_examples.append({"context": normed_ctx, "target": normed_tgt})

        # Normalize query
        query_scaler = StandardScaler()
        initial_query_context = trace[: config.start_context_length]
        normed_query_ctx = query_scaler.fit_transform(
            initial_query_context.reshape(-1, 1)
        ).squeeze()

        current_query = normed_query_ctx.copy()
        predictions = [initial_query_context]

        # Autoregressive prediction
        while len(np.concatenate(predictions)) < trace_length:
            # Format ICL context
            icl_segments = []
            for ex_norm in normalized_examples:
                icl_segments.append(ex_norm["context"])
                icl_segments.append(ex_norm["target"])
            icl_segments.append(current_query)
            icl_context = np.concatenate(icl_segments)

            median_forecast = np.asarray(
                predict_fn(icl_context.astype(np.float32), config.model_prediction_length)
            ).squeeze()

            # Denormalize
            denormed_pred = query_scaler.inverse_transform(
                median_forecast.reshape(-1, 1)
            ).squeeze()
            predictions.append(denormed_pred)

            # Update context
            extended_denormed = np.concatenate(predictions)
            current_query = query_scaler.transform(
                extended_denormed.reshape(-1, 1)
            ).squeeze()

        return np.concatenate(predictions)[:trace_length]

    return forecast_fn


########################################################
# Result models
########################################################


class TraceResult(BaseModel):
    """Per-trace evaluation record (tail means, last 80 timesteps)."""

    trace_key: str
    true_tail_mean: float
    pred_tail_mean: float
    error: float  # pred - true
    abs_error: float
    squared_error: float


class SplitSeedMetrics(BaseModel):
    """Per-seed RMSE for one split (6 ID / 5 OOD tail-mean scalars)."""

    rmse: float
    se_rmse: float


class SeedResult(BaseModel):
    """All records for one example-selection seed."""

    seed: int
    example_ids: dict[str, list[int]]  # trace_key -> raw ids of selected examples
    in_distribution: SplitSeedMetrics
    out_of_distribution: SplitSeedMetrics
    per_trace: list[TraceResult]


class SplitSummary(BaseModel):
    """Cross-seed summary for one split.

    ``rmse``/``se_rmse``/``n_samples`` keep the exact semantics of the legacy
    schema for n_seeds=1; for multi-seed runs they are means over per-seed
    values. ``rmse_std_seeds`` is None for single-seed runs.
    """

    rmse: float
    se_rmse: float
    n_samples: int
    rmse_std_seeds: float | None = None
    rmse_min: float | None = None
    rmse_max: float | None = None


class FewShotRunResults(BaseModel):
    """Full result of one run_benchmark call (superset of the legacy schema)."""

    timestamp: str
    method: str
    config: dict
    seeds: list[int]
    n_seeds: int
    deterministic: bool = False
    in_distribution: SplitSummary
    out_of_distribution: SplitSummary
    per_seed: list[SeedResult]


class PairedComparison(BaseModel):
    """Wilcoxon + paired-bootstrap comparison between two runs (a vs b)."""

    split: str
    pairing: str
    n_pairs: int
    rmse_a: float
    rmse_b: float
    rmse_diff: float  # a - b; negative means a is better
    wilcoxon_statistic: float | None
    wilcoxon_p: float | None
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_p: float
    n_bootstrap: int


########################################################
# Benchmark runner
########################################################


def run_benchmark(
    forecast_fn: ForecastFn,
    config: FewShotConfig,
    example_pool: list[FewShotExample],
    method: str,
    select_fn: SelectFn | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    deterministic: bool = False,
    provider: BenchmarkDataProvider | None = None,
    save_dir: Path | None = None,
    save: bool = True,
) -> FewShotRunResults:
    """Evaluate a forecast function over seeds x benchmark traces.

    Args:
        forecast_fn: Full-trace forecast function (see ForecastFn).
        config: Few-shot configuration (k_shot, context/tail lengths, ...).
        example_pool: Candidate examples (pass the FIXED pool built with
            ``exclude_ids=ID_TEST_RAW_IDS``).
        method: Label for the selection strategy / baseline (used in the
            results JSON and filename).
        select_fn: Example-selection strategy; defaults to seeded random.
        seeds: Example-selection seeds; one full pass over all 11 traces per
            seed.
        deterministic: For runs without stochastic example selection
            (baselines, k=0): collapses to a single pass with no examples.
        provider: Benchmark data provider (created if omitted).
        save_dir: Output directory (default: results/few_shot_v2/).
        save: Whether to write the results JSON.

    Returns:
        FewShotRunResults with per-seed, per-trace records and summaries.
    """
    provider = provider or BenchmarkDataProvider()
    select_fn = select_fn or _select_random
    if deterministic:
        seeds = seeds[:1]

    seed_results: list[SeedResult] = []
    for seed in seeds:
        example_ids: dict[str, list[int]] = {}
        per_trace: list[TraceResult] = []
        split_metrics: dict[str, SplitSeedMetrics] = {}
        for split_name, trace_keys, getter in (
            ("in_distribution", IN_DISTRIBUTION_ITERATIONS, provider.get_id),
            ("out_of_distribution", OUT_OF_DISTRIBUTION_ITERATIONS, provider.get_ood),
        ):
            true_tails: list[float] = []
            pred_tails: list[float] = []
            for trace_key in trace_keys:
                trace = getter(trace_key).numpy()
                query_context = trace[: config.start_context_length]
                if deterministic or config.k_shot == 0:
                    examples: list[FewShotExample] = []
                else:
                    examples = select_fn(example_pool, config.k_shot, seed, query_context)
                example_ids[trace_key] = [ex.trace_id for ex in examples]

                forecast = forecast_fn(trace, examples, config)
                true_tail = float(np.mean(trace[-config.relevant_prediction_tail :]))
                pred_tail = float(np.mean(forecast[-config.relevant_prediction_tail :]))
                error = pred_tail - true_tail
                per_trace.append(
                    TraceResult(
                        trace_key=trace_key,
                        true_tail_mean=true_tail,
                        pred_tail_mean=pred_tail,
                        error=error,
                        abs_error=abs(error),
                        squared_error=error**2,
                    )
                )
                true_tails.append(true_tail)
                pred_tails.append(pred_tail)
            rmse, se_rmse = rmse_with_standard_error(
                np.array(true_tails), np.array(pred_tails)
            )
            split_metrics[split_name] = SplitSeedMetrics(rmse=float(rmse), se_rmse=float(se_rmse))

        seed_results.append(
            SeedResult(
                seed=seed,
                example_ids=example_ids,
                in_distribution=split_metrics["in_distribution"],
                out_of_distribution=split_metrics["out_of_distribution"],
                per_trace=per_trace,
            )
        )

    def summarize(split_name: str, n_samples: int) -> SplitSummary:
        rmses = np.array([getattr(sr, split_name).rmse for sr in seed_results])
        ses = np.array([getattr(sr, split_name).se_rmse for sr in seed_results])
        return SplitSummary(
            rmse=float(np.mean(rmses)),
            se_rmse=float(np.mean(ses)),
            n_samples=n_samples,
            rmse_std_seeds=float(np.std(rmses, ddof=1)) if len(rmses) > 1 else None,
            rmse_min=float(np.min(rmses)),
            rmse_max=float(np.max(rmses)),
        )

    results = FewShotRunResults(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        method=method,
        config=config.model_dump(),
        seeds=list(seeds),
        n_seeds=len(seeds),
        deterministic=deterministic,
        in_distribution=summarize("in_distribution", len(IN_DISTRIBUTION_ITERATIONS)),
        out_of_distribution=summarize(
            "out_of_distribution", len(OUT_OF_DISTRIBUTION_ITERATIONS)
        ),
        per_seed=seed_results,
    )

    if save:
        save_dir = Path(save_dir) if save_dir is not None else DEFAULT_SAVE_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            save_dir
            / f"{results.timestamp}_{method}_k{config.k_shot}_fewshot_results.json"
        )
        with open(out_path, "w") as f:
            json.dump(results.model_dump(), f, indent=2)
        print(f"Results saved to: {out_path}")

    return results


########################################################
# Significance testing
########################################################


def _per_pair_squared_errors(
    results: FewShotRunResults | dict,
    split: Literal["in_distribution", "out_of_distribution"],
    pairing: Literal["trace", "trace_seed"],
) -> dict[tuple[int, str] | str, float]:
    """Extract squared errors keyed by pairing unit from a results object."""
    data = results.model_dump() if isinstance(results, BaseModel) else results
    is_ood = split == "out_of_distribution"
    raw: dict[tuple[int, str], float] = {}
    for seed_result in data["per_seed"]:
        for tr in seed_result["per_trace"]:
            if tr["trace_key"].startswith("ood_") == is_ood:
                raw[(seed_result["seed"], tr["trace_key"])] = tr["squared_error"]
    if pairing == "trace_seed":
        return raw
    # pairing == "trace": average squared errors over seeds per trace
    by_trace: dict[str, list[float]] = {}
    for (_, trace_key), sq in raw.items():
        by_trace.setdefault(trace_key, []).append(sq)
    return {trace_key: float(np.mean(values)) for trace_key, values in by_trace.items()}


def paired_comparison(
    results_a: FewShotRunResults | dict,
    results_b: FewShotRunResults | dict,
    split: Literal["in_distribution", "out_of_distribution"],
    pairing: Literal["trace", "trace_seed"] = "trace",
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PairedComparison:
    """Paired significance test between two runs on the same benchmark traces.

    Wilcoxon signed-rank over paired squared errors (seed-averaged per trace
    for ``pairing="trace"``; all (trace, seed) pairs for ``"trace_seed"``),
    plus a paired bootstrap over the pairing units giving a 95% CI and
    two-sided p for the RMSE difference (a - b).

    CAVEAT — Wilcoxon power: with only 6 ID / 5 OOD traces the smallest
    achievable two-sided p under ``pairing="trace"`` is 0.031 (n=6) / 0.0625
    (n=5), so OOD differences can never reach p<0.05 by Wilcoxon alone. Treat
    the bootstrap CI as the primary evidence, especially on OOD;
    ``pairing="trace_seed"`` adds resolution but treats seeds as independent.

    Args:
        results_a: First run (FewShotRunResults or its dict/JSON form).
        results_b: Second run, same traces (and same seeds for "trace_seed").
        split: Which benchmark split to compare.
        pairing: Pairing unit for the tests.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for the bootstrap.

    Returns:
        PairedComparison with both test results. ``rmse_diff < 0`` means
        ``results_a`` is better.
    """
    sq_a = _per_pair_squared_errors(results_a, split, pairing)
    sq_b = _per_pair_squared_errors(results_b, split, pairing)
    if set(sq_a) != set(sq_b):
        raise ValueError(
            f"Pairing units differ between runs: {sorted(set(sq_a) ^ set(sq_b))}"
        )
    keys = sorted(sq_a)
    a = np.array([sq_a[key] for key in keys])
    b = np.array([sq_b[key] for key in keys])
    diffs = a - b

    if np.allclose(diffs, 0.0):
        wilcoxon_statistic, wilcoxon_p = None, 1.0
    else:
        try:
            statistic, p_value = stats.wilcoxon(diffs)
            wilcoxon_statistic, wilcoxon_p = float(statistic), float(p_value)
        except ValueError:
            wilcoxon_statistic, wilcoxon_p = None, None

    rmse_a = float(np.sqrt(np.mean(a)))
    rmse_b = float(np.sqrt(np.mean(b)))

    rng = np.random.default_rng(seed)
    n = len(keys)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_diffs = np.sqrt(np.mean(a[indices], axis=1)) - np.sqrt(np.mean(b[indices], axis=1))
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    p_low = float(np.mean(boot_diffs <= 0.0))
    p_high = float(np.mean(boot_diffs >= 0.0))
    bootstrap_p = min(1.0, 2.0 * min(p_low, p_high))

    return PairedComparison(
        split=split,
        pairing=pairing,
        n_pairs=n,
        rmse_a=rmse_a,
        rmse_b=rmse_b,
        rmse_diff=rmse_a - rmse_b,
        wilcoxon_statistic=wilcoxon_statistic,
        wilcoxon_p=wilcoxon_p,
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        bootstrap_p=bootstrap_p,
        n_bootstrap=n_bootstrap,
    )


########################################################
# Aggregation
########################################################


def load_results(
    results_dir: Path | str,
    pattern: str = "*_fewshot_results.json",
) -> list[dict]:
    """Load result JSONs from a directory, normalizing legacy files.

    Legacy files (pre-harness, e.g. results/few_shot_t{80,266}) lack the
    method/seeds fields; the method is recovered from the filename and
    ``n_seeds`` is set to 1.
    """
    results: list[dict] = []
    for path in sorted(Path(results_dir).glob(pattern)):
        data = json.load(open(path, "r"))
        data["_path"] = str(path)
        if "method" not in data:
            match = LEGACY_FILENAME_RE.match(path.stem)
            data["method"] = match.group("method") if match else path.stem
            data["seeds"] = [data.get("config", {}).get("random_seed", 42)]
            data["n_seeds"] = 1
        results.append(data)
    return results


def results_table(results: list[dict]) -> pd.DataFrame:
    """Aggregate loaded results into a comparison table."""

    def std_or_nan(split: dict) -> float:
        value = split.get("rmse_std_seeds")
        return float("nan") if value is None else float(value)

    rows = []
    for r in results:
        config = r.get("config", {})
        rows.append(
            {
                "model": config.get("model_slug"),
                "method": r.get("method"),
                "k": config.get("k_shot"),
                "n_seeds": r.get("n_seeds", 1),
                "id_rmse": r["in_distribution"]["rmse"],
                "id_se": r["in_distribution"]["se_rmse"],
                "id_rmse_std": std_or_nan(r["in_distribution"]),
                "ood_rmse": r["out_of_distribution"]["rmse"],
                "ood_se": r["out_of_distribution"]["se_rmse"],
                "ood_rmse_std": std_or_nan(r["out_of_distribution"]),
                "timestamp": r.get("timestamp"),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["model", "method", "k"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from fusiontimeseries.benchmarking.few_shot.few_shot_utils import create_example_pool
    from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS

    print("Harness smoke test (CPU, constant forecast)...")

    config = FewShotConfig(
        device="cpu",
        model_slug="smoke-test/constant",
        model_prediction_length=64,
        start_context_length=80,
        relevant_prediction_tail=80,
        k_shot=2,
        random_seed=42,
        example_target_length=None,
    )
    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)

    def constant_forecast(
        trace: NDArray[np.float32],
        examples: list[FewShotExample],
        config: FewShotConfig,
    ) -> NDArray[np.float32]:
        context = trace[: config.start_context_length]
        level = np.full(len(trace) - len(context), context[-1], dtype=trace.dtype)
        return np.concatenate([context, level])

    results = run_benchmark(
        forecast_fn=constant_forecast,
        config=config,
        example_pool=pool,
        method="smoke_constant",
        seeds=(0, 1),
        save=False,
    )
    assert results.n_seeds == 2
    assert all(len(sr.per_trace) == 11 for sr in results.per_seed)
    assert results.per_seed[0].in_distribution.rmse == results.per_seed[1].in_distribution.rmse, (
        "Constant forecast must be seed-invariant"
    )
    print(
        f"✓ run_benchmark: ID {results.in_distribution.rmse:.2f} ± "
        f"{results.in_distribution.se_rmse:.2f}, "
        f"OOD {results.out_of_distribution.rmse:.2f} ± "
        f"{results.out_of_distribution.se_rmse:.2f}"
    )

    # ICL rollout with a trivial PredictFn: shapes + context passthrough
    def last_value_predict(context: NDArray[np.float32], prediction_length: int) -> NDArray[np.float32]:
        return np.full(prediction_length, context[-1], dtype=np.float32)

    icl_fn = make_icl_forecast_fn(last_value_predict)
    provider = BenchmarkDataProvider()
    trace = provider.get_id("iteration_8_ifft").numpy()
    examples = select_examples_random(pool, k=2, seed=0)
    forecast = icl_fn(trace, examples, config)
    assert forecast.shape == trace.shape
    assert np.allclose(forecast[: config.start_context_length], trace[: config.start_context_length])
    forecast_zero_shot = icl_fn(trace, [], config)
    assert forecast_zero_shot.shape == trace.shape
    print("✓ make_icl_forecast_fn: rollout shapes OK (k=2 and k=0)")

    # Self-comparison: CI must contain 0
    comparison = paired_comparison(results, results, "in_distribution")
    assert comparison.bootstrap_ci_low <= 0.0 <= comparison.bootstrap_ci_high
    assert comparison.rmse_diff == 0.0 and comparison.wilcoxon_p == 1.0
    print(
        f"✓ paired_comparison self-test: diff={comparison.rmse_diff:.3f}, "
        f"CI=[{comparison.bootstrap_ci_low:.3f}, {comparison.bootstrap_ci_high:.3f}], "
        f"wilcoxon_p={comparison.wilcoxon_p}"
    )

    # Legacy results load + table
    legacy_dir = REPO_ROOT / "results" / "few_shot_t266"
    legacy = load_results(legacy_dir)
    assert len(legacy) > 0, f"No legacy results found in {legacy_dir}"
    table = results_table(legacy)
    assert table["n_seeds"].eq(1).all() and table["id_rmse_std"].isna().all()
    print(f"✓ load_results/results_table: {len(legacy)} legacy files from {legacy_dir.name}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n✅ Harness smoke test passed!")
