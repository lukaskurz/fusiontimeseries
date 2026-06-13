"""Retrieval-based example selection for few-shot ICL (Phase 2).

Replaces random example selection with informed retrieval. Strategies (the
``STRATEGIES`` ids are used as ``method`` suffixes in result files and plots):

- ``random``: seeded random selection (Phase-1 default, the control).
- ``op_knn``: k nearest in normalized operating-parameter space (q, shat,
  rlt, rln); requires the Phase-0 mapping and filters the one pool example
  with ``operating_params=None`` (244 of 245 candidates).
- ``ctx_euclid`` / ``ctx_dtw`` / ``ctx_growth``: k nearest by query-context
  similarity — Euclidean on z-scored contexts, DTW on z-scored contexts, or
  absolute difference of linear-phase growth rates (physics-motivated:
  theory links growth rate to saturation amplitude).
- ``ctx_level``: k nearest by ABSOLUTE context level — ``|mean(ctx_q) -
  mean(ctx_i)|`` on RAW (un-z-scored) contexts. The other ``ctx_*`` strategies
  z-score each context, which erases exactly the absolute level the tail-mean
  metric scores; ``ctx_level`` matches on that level instead of shape. The
  early-context mean is the strongest context-side predictor of the saturation
  level (Phase-7 mechanism analysis: ρ ≈ +0.89 with the trace's own tail), so
  this is the level-matching counterpart to the shape-matching retrievers
  (model-free demonstration in ``analyze_level_matching.py``).
- ``oracle_tail``: CHEATING DIAGNOSTIC — selects by the query's ground-truth
  tail mean. Never a method result; only a headroom estimate.
- ``mmr_euclid``: max-marginal-relevance variant of ``ctx_euclid`` trading
  shape similarity against shape diversity among the selected examples.
- ``mmr_level``: level-aware MMR — relevance is context-LEVEL proximity (like
  ``ctx_level``) while redundancy is z-scored SHAPE similarity. Picks examples
  close in absolute level but diverse in shape: the principled "make retrieval
  level-aware while keeping demonstration variety" hybrid.

Ordering convention: every selector returns the MOST SIMILAR example LAST,
i.e. adjacent to the query context in the flat ICL concatenation (recency
bias favours the segment closest to the query; Phase 3 ablates ordering).
``example_ids`` in the results JSON preserves this order.

z-scoring is an exact copy of ``baselines.make_knn_copy_forecast``'s logic so
``ctx_euclid`` retrieves the same neighbours as the kNN-copy baseline (the
self-test cross-checks this).
"""

from functools import cache

import numpy as np
from numpy.typing import NDArray

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotExample,
    select_examples_random,
)
from fusiontimeseries.benchmarking.few_shot.harness import SelectFn
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    OP_NAMES,
    get_params_for_benchmark_trace,
    normalize_params,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    IN_DISTRIBUTION_ITERATIONS,
    OUT_OF_DISTRIBUTION_ITERATIONS,
    BenchmarkDataProvider,
)

__all__ = [
    "STRATEGIES",
    "make_select_fn",
    "dtw_distance",
    "estimate_growth_rate",
    "select_examples_op_knn",
    "select_examples_context_nn",
    "select_examples_oracle",
    "select_examples_mmr",
    "select_examples_level_mmr",
]

STRATEGIES: tuple[str, ...] = (
    "random",
    "op_knn",
    "ctx_euclid",
    "ctx_dtw",
    "ctx_growth",
    "ctx_level",
    "oracle_tail",
    "mmr_euclid",
    "mmr_level",
)

CONTEXT_DISTANCES: dict[str, str] = {
    "ctx_euclid": "euclidean",
    "ctx_dtw": "dtw",
    "ctx_growth": "growth_rate",
    "ctx_level": "level",
}


def _zscore(x: NDArray) -> NDArray[np.float64]:
    """Per-sample z-score (exact copy of the kNN-copy baseline's logic)."""
    x = np.asarray(x, dtype=np.float64)
    mean, std = float(x.mean()), float(x.std())
    std = std if std > 0 else 1.0
    return (x - mean) / std


def dtw_distance(
    a: NDArray, b: NDArray, band: int | None = None
) -> float:
    """Dynamic-time-warping distance with absolute-difference local cost.

    O(n*m) dynamic program; the local cost row ``|a[i-1] - b|`` is vectorized
    per row. An optional Sakoe-Chiba ``band`` restricts warping to
    ``|i - j| <= band``.

    Args:
        a: First series (1D).
        b: Second series (1D).
        band: Sakoe-Chiba band half-width in steps (None = unconstrained).

    Returns:
        Accumulated warping cost (0.0 for identical series).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = 1 if band is None else max(1, i - band)
        hi = m if band is None else min(m, i + band)
        cost_row = np.abs(a[i - 1] - b[lo - 1 : hi])
        for j in range(lo, hi + 1):
            acc[i, j] = cost_row[j - lo] + min(
                acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
            )
    return float(acc[n, m])


def estimate_growth_rate(
    context: NDArray, clamp: float = 1e-3, min_fit_points: int = 16
) -> float:
    """Linear-phase growth rate: log-linear slope up to the overshoot peak.

    Fits ``log(clip(context, clamp))`` over ``[0 : argmax(context) + 1]`` by
    least squares — exponential growth ends at the overshoot peak. If the
    peak comes earlier than ``min_fit_points`` steps (early-peak/noisy
    contexts), falls back to fitting the full context.

    Args:
        context: Raw (NOT z-scored) flux context.
        clamp: Lower clip before the log (guards zeros/negatives).
        min_fit_points: Minimum window length for the peak-bounded fit.

    Returns:
        Least-squares slope of the log-flux (per subsampled timestep).
    """
    ctx = np.asarray(context, dtype=np.float64)
    end = int(np.argmax(ctx)) + 1
    window = ctx[:end] if end >= min_fit_points else ctx
    log_flux = np.log(np.clip(window, clamp, None))
    t = np.arange(len(log_flux), dtype=np.float64)
    slope = np.polyfit(t, log_flux, 1)[0]
    return float(slope)


########################################################
# Rankings (full pool order, nearest FIRST)
########################################################


def _params_vector(params: dict[str, float]) -> NDArray[np.float64]:
    """Normalized (min-max) operating-parameter vector in OP_NAMES order."""
    normalized = normalize_params(params)
    return np.array([normalized[name] for name in OP_NAMES], dtype=np.float64)


def _rank_op_knn(pool: list[FewShotExample], query_key: str) -> list[int]:
    """Pool indices ranked by operating-param distance (nearest first).

    Pool examples without operating params are excluded from the ranking.
    """
    query_vec = _params_vector(get_params_for_benchmark_trace(query_key))
    candidates = [i for i, ex in enumerate(pool) if ex.operating_params is not None]
    dists = np.array(
        [
            np.linalg.norm(_params_vector(pool[i].operating_params) - query_vec)
            for i in candidates
        ]
    )
    order = np.argsort(dists, kind="stable")
    return [candidates[j] for j in order]


def _rank_context_nn(
    pool: list[FewShotExample],
    query_context: NDArray,
    distance: str,
    band: int | None = None,
) -> list[int]:
    """Pool indices ranked by context similarity (nearest first)."""
    if distance == "euclidean":
        query_z = _zscore(query_context)
        pool_matrix = np.stack([_zscore(ex.context_array) for ex in pool])
        dists = np.linalg.norm(pool_matrix - query_z, axis=1)
    elif distance == "dtw":
        query_z = _zscore(query_context)
        dists = np.array(
            [dtw_distance(query_z, _zscore(ex.context_array), band=band) for ex in pool]
        )
    elif distance == "growth_rate":
        query_gamma = estimate_growth_rate(query_context)
        dists = np.array(
            [abs(estimate_growth_rate(ex.context_array) - query_gamma) for ex in pool]
        )
    elif distance == "level":
        # Absolute context LEVEL: raw (NOT z-scored) context means. Matches the
        # signal the tail-mean metric scores, which every z-scored distance
        # above erases.
        query_level = float(np.mean(query_context))
        dists = np.array(
            [abs(float(np.mean(ex.context_array)) - query_level) for ex in pool]
        )
    else:
        raise ValueError(f"Unknown context distance: {distance}")
    return list(np.argsort(dists, kind="stable"))


@cache
def _benchmark_true_tail_means(tail: int = 80) -> dict[str, float]:
    """Ground-truth tail means of all 11 benchmark traces (oracle labels)."""
    provider = BenchmarkDataProvider()
    means: dict[str, float] = {}
    for key in IN_DISTRIBUTION_ITERATIONS:
        means[key] = float(np.mean(provider.get_id(key).numpy()[-tail:]))
    for key in OUT_OF_DISTRIBUTION_ITERATIONS:
        means[key] = float(np.mean(provider.get_ood(key).numpy()[-tail:]))
    return means


def _rank_oracle(
    pool: list[FewShotExample], query_key: str, tail: int = 80
) -> list[int]:
    """Pool indices ranked by |example tail mean - TRUE query tail mean|."""
    true_mean = _benchmark_true_tail_means(tail)[query_key]
    pool_tail_means = np.array(
        [float(np.mean(ex.trace_array[-tail:])) for ex in pool]
    )
    return list(np.argsort(np.abs(pool_tail_means - true_mean), kind="stable"))


def _top_k_nearest_last(
    pool: list[FewShotExample], ranked: list[int], k: int
) -> list[FewShotExample]:
    """Top-k of a nearest-first ranking, reordered so the nearest comes LAST."""
    if k > len(ranked):
        raise ValueError(f"Cannot select {k} examples from {len(ranked)} candidates")
    return [pool[i] for i in reversed(ranked[:k])]


########################################################
# Public selectors
########################################################


def select_examples_op_knn(
    pool: list[FewShotExample], k: int, query_key: str
) -> list[FewShotExample]:
    """k nearest pool examples in normalized operating-parameter space.

    Distance is Euclidean over the min-max-normalized (q, shat, rlt, rln)
    vectors; pool examples without operating params (1 of 245) are filtered.

    Args:
        pool: Example pool (FIXED pool, test ids excluded).
        k: Number of examples.
        query_key: Benchmark trace key (e.g. ``"iteration_8_ifft"``).

    Returns:
        k examples, most similar LAST (adjacent to the query).
    """
    return _top_k_nearest_last(pool, _rank_op_knn(pool, query_key), k)


def select_examples_context_nn(
    pool: list[FewShotExample],
    k: int,
    query_context: NDArray,
    distance: str = "euclidean",
    band: int | None = None,
) -> list[FewShotExample]:
    """k nearest pool examples by query-context similarity.

    Args:
        pool: Example pool (FIXED pool, test ids excluded).
        k: Number of examples.
        query_context: Raw 80-step query context.
        distance: ``"euclidean"`` (z-scored contexts, matches the kNN-copy
            baseline's neighbours), ``"dtw"`` (z-scored contexts),
            ``"growth_rate"`` (|gamma_example - gamma_query| on raw contexts),
            or ``"level"`` (|mean(ctx) - mean(query)| on raw contexts — matches
            absolute level, not shape).
        band: Sakoe-Chiba band for DTW (None = unconstrained).

    Returns:
        k examples, most similar LAST (adjacent to the query).
    """
    return _top_k_nearest_last(
        pool, _rank_context_nn(pool, query_context, distance, band=band), k
    )


def select_examples_oracle(
    pool: list[FewShotExample], k: int, query_key: str, tail: int = 80
) -> list[FewShotExample]:
    """CHEATING DIAGNOSTIC: select by the query's ground-truth tail mean.

    Picks the k pool examples whose own tail means (last ``tail`` steps, the
    same definition as the benchmark metric) are closest to the query's TRUE
    tail mean. This reads the test label and is NEVER a legitimate method —
    it estimates the headroom of label-aware selection under the current
    presentation format. (It is nearest-LABEL selection, not a true
    model-in-the-loop upper bound; a greedy model-in-the-loop oracle is out
    of scope at ~10 h/model.)

    Args:
        pool: Example pool (FIXED pool, test ids excluded).
        k: Number of examples.
        query_key: Benchmark trace key.
        tail: Tail window (must match ``relevant_prediction_tail``).

    Returns:
        k examples, closest tail mean LAST (adjacent to the query).
    """
    return _top_k_nearest_last(pool, _rank_oracle(pool, query_key, tail=tail), k)


def select_examples_mmr(
    pool: list[FewShotExample],
    k: int,
    query_context: NDArray,
    lambda_: float = 0.5,
) -> list[FewShotExample]:
    """Max-marginal-relevance selection: similarity traded against diversity.

    Greedy MMR on z-scored Euclidean distances with ``sim = 1 / (1 + dist)``:
    each step picks ``argmax lambda_ * sim(query, e) - (1 - lambda_) *
    max(sim(e, s) for s in selected)``. The first pick is always the plain
    nearest neighbour; the returned list is reversed so that this
    highest-relevance pick sits LAST, adjacent to the query.

    Args:
        pool: Example pool (FIXED pool, test ids excluded).
        k: Number of examples.
        query_context: Raw 80-step query context.
        lambda_: Relevance weight (1.0 reduces to plain ctx_euclid top-k).

    Returns:
        k examples, most query-similar pick LAST.
    """
    pool_matrix = np.stack([_zscore(ex.context_array) for ex in pool])
    query_z = _zscore(query_context)
    sim_query = 1.0 / (1.0 + np.linalg.norm(pool_matrix - query_z, axis=1))

    selected: list[int] = []
    remaining = list(range(len(pool)))
    if k > len(pool):
        raise ValueError(f"Cannot select {k} examples from pool of size {len(pool)}")
    while len(selected) < k:
        best_index, best_score = -1, -np.inf
        for i in remaining:
            if selected:
                sims = 1.0 / (
                    1.0 + np.linalg.norm(pool_matrix[selected] - pool_matrix[i], axis=1)
                )
                redundancy = float(np.max(sims))
            else:
                redundancy = 0.0
            score = lambda_ * float(sim_query[i]) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        remaining.remove(best_index)
    return [pool[i] for i in reversed(selected)]


def select_examples_level_mmr(
    pool: list[FewShotExample],
    k: int,
    query_context: NDArray,
    lambda_: float = 0.5,
) -> list[FewShotExample]:
    """Level-aware MMR: level relevance traded against shape diversity.

    The level-matching counterpart of ``select_examples_mmr``. Relevance is
    absolute-level proximity (the ``ctx_level`` signal — the raw context mean,
    the strongest context-side predictor of the saturation level, Phase-7
    ρ ≈ +0.89), while redundancy is SHAPE similarity (``1 / (1 + ||z(ctx_i) -
    z(ctx_j)||)`` on z-scored contexts, the ``ctx_euclid`` distance) among the
    already-selected. Each step picks ``argmax lambda_ * sim_level(query, e) -
    (1 - lambda_) * max(sim_shape(e, s) for s in selected)``: examples close in
    LEVEL but diverse in SHAPE. ``ctx_level`` matches level but ignores shape,
    so its picks can be near-duplicate trajectories; this keeps the
    demonstration set varied without sacrificing the level signal the tail-mean
    metric scores.

    The level relevance is ``sim_level = 1 / (1 + |Δlevel| / s)`` where the raw
    level gap ``|Δlevel|`` is divided by the pool-level spread ``s = std(pool
    context means)``. Without this normalization the two similarities live on
    incomparable scales — raw flux-unit level gaps vs unitless z-scored shape
    distances — and the level term swamps the shape term at ``lambda_ = 0.5``,
    collapsing the strategy back to plain ``ctx_level``. Standardizing the
    level gap to pool-spread units makes the relevance dynamic range comparable
    to the shape redundancy, so ``lambda_`` genuinely trades the two off
    (empirically the picks then differ from ``ctx_level`` and are more
    shape-diverse for 9/11 benchmark queries). ``s`` depends only on the pool,
    so the selection stays deterministic per ``(pool, query)``.

    The first pick is always the plain ``ctx_level`` nearest neighbour
    (redundancy 0); the returned list is reversed so it sits LAST, adjacent to
    the query.

    Args:
        pool: Example pool (FIXED pool, test ids excluded).
        k: Number of examples.
        query_context: Raw 80-step query context.
        lambda_: Level-relevance weight (1.0 reduces to plain ctx_level top-k).

    Returns:
        k examples, most level-relevant pick LAST.
    """
    if k > len(pool):
        raise ValueError(f"Cannot select {k} examples from pool of size {len(pool)}")
    query_level = float(np.mean(query_context))
    pool_levels = np.array([float(np.mean(ex.context_array)) for ex in pool])
    level_scale = float(np.std(pool_levels))
    level_scale = level_scale if level_scale > 0 else 1.0
    sim_query = 1.0 / (1.0 + np.abs(pool_levels - query_level) / level_scale)
    pool_matrix = np.stack([_zscore(ex.context_array) for ex in pool])

    selected: list[int] = []
    remaining = list(range(len(pool)))
    while len(selected) < k:
        best_index, best_score = -1, -np.inf
        for i in remaining:
            if selected:
                sims = 1.0 / (
                    1.0 + np.linalg.norm(pool_matrix[selected] - pool_matrix[i], axis=1)
                )
                redundancy = float(np.max(sims))
            else:
                redundancy = 0.0
            score = lambda_ * float(sim_query[i]) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        remaining.remove(best_index)
    return [pool[i] for i in reversed(selected)]


########################################################
# SelectFn registry
########################################################


def make_select_fn(strategy: str) -> SelectFn:
    """Build a harness-compatible SelectFn for a strategy id.

    For ranking-based strategies the full distance ranking is memoized per
    ``(pool, query_key)`` inside the returned closure, so expensive distances
    (DTW) run once per query across all k values and models — build the
    SelectFns ONCE and reuse them across the whole grid.

    Args:
        strategy: One of ``STRATEGIES``.

    Returns:
        A SelectFn ``(pool, k, seed, query_context, query_key) -> examples``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    if strategy == "random":

        def select_random(
            pool: list[FewShotExample],
            k: int,
            seed: int,
            query_context: NDArray[np.float32],
            query_key: str,
        ) -> list[FewShotExample]:
            return select_examples_random(pool, k=k, seed=seed)

        return select_random

    if strategy == "mmr_euclid":

        def select_mmr(
            pool: list[FewShotExample],
            k: int,
            seed: int,
            query_context: NDArray[np.float32],
            query_key: str,
        ) -> list[FewShotExample]:
            return select_examples_mmr(pool, k, query_context)

        return select_mmr

    if strategy == "mmr_level":

        def select_mmr_level(
            pool: list[FewShotExample],
            k: int,
            seed: int,
            query_context: NDArray[np.float32],
            query_key: str,
        ) -> list[FewShotExample]:
            return select_examples_level_mmr(pool, k, query_context)

        return select_mmr_level

    ranking_cache: dict[tuple[int, str], list[int]] = {}

    def select_ranked(
        pool: list[FewShotExample],
        k: int,
        seed: int,
        query_context: NDArray[np.float32],
        query_key: str,
    ) -> list[FewShotExample]:
        cache_key = (id(pool), query_key)
        if cache_key not in ranking_cache:
            if strategy == "op_knn":
                ranking_cache[cache_key] = _rank_op_knn(pool, query_key)
            elif strategy == "oracle_tail":
                ranking_cache[cache_key] = _rank_oracle(pool, query_key)
            else:
                ranking_cache[cache_key] = _rank_context_nn(
                    pool, query_context, CONTEXT_DISTANCES[strategy]
                )
        return _top_k_nearest_last(pool, ranking_cache[cache_key], k)

    return select_ranked


if __name__ == "__main__":
    from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
        FewShotConfig,
        create_example_pool,
    )
    from fusiontimeseries.benchmarking.few_shot.harness import run_benchmark
    from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS

    print("Selection self-tests (Phase 2)...")

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    assert len(pool) == 245, f"Expected fixed pool of 245, got {len(pool)}"
    provider = BenchmarkDataProvider()

    ########################################################
    # op_knn
    ########################################################
    with_params = [ex for ex in pool if ex.operating_params is not None]
    assert len(with_params) == 244, f"Expected 244 examples with params, got {len(with_params)}"
    picks_a = select_examples_op_knn(pool, 5, "iteration_8_ifft")
    picks_b = select_examples_op_knn(pool, 5, "iteration_8_ifft")
    assert [e.trace_id for e in picks_a] == [e.trace_id for e in picks_b], (
        "op_knn must be deterministic"
    )
    assert all(e.trace_id not in ID_TEST_RAW_IDS for e in picks_a)
    assert all(e.operating_params is not None for e in picks_a)
    query_params = get_params_for_benchmark_trace("iteration_8_ifft")
    print(f"✓ op_knn: 244 candidates, deterministic. Query iteration_8_ifft params: {query_params}")
    print("  nearest (LAST) → furthest (FIRST):")
    for ex in reversed(picks_a):
        print(f"    raw {ex.trace_id}: {ex.operating_params}")

    ########################################################
    # ctx_euclid == kNN-copy neighbours
    ########################################################
    pool_matrix = np.stack([_zscore(ex.context_array) for ex in pool])
    for key in IN_DISTRIBUTION_ITERATIONS:
        trace = provider.get_id(key).numpy()
        query_context = trace[:80]
        # kNN-copy's neighbour computation (baselines.make_knn_copy_forecast)
        distances = np.linalg.norm(pool_matrix - _zscore(query_context), axis=1)
        knn_neighbours = np.argsort(distances, kind="stable")[:5]
        selected = select_examples_context_nn(pool, 5, query_context, distance="euclidean")
        assert [ex.trace_id for ex in reversed(selected)] == [
            pool[i].trace_id for i in knn_neighbours
        ], f"ctx_euclid k=5 != kNN-copy neighbours for {key}"
    print("✓ ctx_euclid: k=5 picks match kNN-copy's 5 nearest for all 6 ID traces")

    ########################################################
    # DTW
    ########################################################
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    assert dtw_distance(a, a) == 0.0, "DTW identity failed"
    assert np.isclose(dtw_distance(a, b), dtw_distance(b, a)), "DTW symmetry failed"
    assert dtw_distance(a, b) <= np.sum(np.abs(a - b)) + 1e-9, (
        "DTW must not exceed the diagonal-path (L1) cost"
    )
    assert np.isclose(dtw_distance(a, b, band=len(a)), dtw_distance(a, b)), (
        "Full-width band must equal unconstrained DTW"
    )
    t = np.arange(80)
    sine = np.sin(2 * np.pi * t / 32)
    shifted = np.sin(2 * np.pi * (t - 8) / 32)  # quarter-period shift
    flat = np.zeros(80)
    eu_shift = float(np.linalg.norm(_zscore(sine) - _zscore(shifted)))
    eu_flat = float(np.linalg.norm(_zscore(sine) - _zscore(flat)))
    dtw_shift = dtw_distance(_zscore(sine), _zscore(shifted))
    dtw_flat = dtw_distance(_zscore(sine), _zscore(flat))
    assert eu_flat < eu_shift, "Euclidean should prefer flat over shifted sine"
    assert dtw_shift < dtw_flat, "DTW should prefer shifted sine over flat"
    print(
        f"✓ dtw_distance: identity/symmetry/L1-bound OK; shifted sine flips rank "
        f"(euclid {eu_shift:.1f} vs {eu_flat:.1f}; dtw {dtw_shift:.1f} vs {dtw_flat:.1f})"
    )

    ########################################################
    # Growth rate
    ########################################################
    gamma_true = 0.08
    tt = np.arange(80, dtype=np.float64)
    synth = 0.05 * np.exp(gamma_true * np.minimum(tt, 60))
    synth *= np.exp(rng.normal(0.0, 0.05, size=80))  # 5% multiplicative noise
    gamma_est = estimate_growth_rate(synth)
    assert abs(gamma_est - gamma_true) / gamma_true < 0.1, (
        f"Growth-rate estimate {gamma_est:.4f} off >10% from {gamma_true}"
    )
    pool_gammas = np.array([estimate_growth_rate(ex.context_array) for ex in pool])
    assert np.all(np.isfinite(pool_gammas)), "Non-finite growth rates in pool"
    print(
        f"✓ estimate_growth_rate: synthetic gamma {gamma_true} -> {gamma_est:.4f}; "
        f"pool gamma min/median/max = {pool_gammas.min():.3f}/"
        f"{np.median(pool_gammas):.3f}/{pool_gammas.max():.3f}"
    )

    ########################################################
    # ctx_level — match by absolute context level, not shape
    ########################################################
    query_context = provider.get_id("iteration_8_ifft").numpy()[:80]
    q_level = float(np.mean(query_context))
    pool_levels = np.array([float(np.mean(ex.context_array)) for ex in pool])
    expected_nn = np.argsort(np.abs(pool_levels - q_level), kind="stable")[:5]
    level_picks = select_examples_context_nn(pool, 5, query_context, distance="level")
    assert [ex.trace_id for ex in reversed(level_picks)] == [
        pool[i].trace_id for i in expected_nn
    ], "ctx_level must rank by |context mean - query mean|"
    # ctx_level and ctx_euclid must generally pick DIFFERENT neighbours (level
    # vs shape are distinct signals) — else the strategy adds nothing.
    euclid_picks = select_examples_context_nn(pool, 5, query_context, distance="euclidean")
    assert {e.trace_id for e in level_picks} != {e.trace_id for e in euclid_picks}, (
        "ctx_level and ctx_euclid should differ (level ≠ shape)"
    )
    print(
        f"✓ ctx_level: ranks by |ctx mean − query mean| (query level {q_level:.1f}); "
        f"picks differ from ctx_euclid (level vs shape)"
    )

    ########################################################
    # Oracle
    ########################################################
    true_means = _benchmark_true_tail_means(80)
    pool_tails = np.array([float(np.mean(ex.trace_array[-80:])) for ex in pool])
    for key, true_mean in true_means.items():
        picked = select_examples_oracle(pool, 1, key)[0]
        expected = pool[int(np.argmin(np.abs(pool_tails - true_mean)))]
        assert picked.trace_id == expected.trace_id, f"Oracle k=1 mismatch for {key}"
    print(f"✓ oracle_tail: k=1 == argmin |tail - true| for all {len(true_means)} benchmark traces")

    ########################################################
    # MMR
    ########################################################
    query_context = provider.get_id("iteration_8_ifft").numpy()[:80]
    mmr_picks = select_examples_mmr(pool, 5, query_context, lambda_=0.5)
    nearest = select_examples_context_nn(pool, 1, query_context)[0]
    assert mmr_picks[-1].trace_id == nearest.trace_id, (
        "MMR's first (most relevant) pick must be the plain nearest neighbour"
    )

    def min_pairwise_dist(examples: list[FewShotExample]) -> float:
        zs = [_zscore(ex.context_array) for ex in examples]
        return min(
            float(np.linalg.norm(zs[i] - zs[j]))
            for i in range(len(zs))
            for j in range(i + 1, len(zs))
        )

    topk_picks = select_examples_context_nn(pool, 5, query_context)
    mmr_diversity = min_pairwise_dist(mmr_picks)
    topk_diversity = min_pairwise_dist(topk_picks)
    assert mmr_diversity > topk_diversity, (
        f"MMR selection not more diverse: {mmr_diversity:.2f} vs {topk_diversity:.2f}"
    )
    print(
        f"✓ mmr_euclid: nearest-neighbour anchor kept; min pairwise dist "
        f"{mmr_diversity:.2f} > top-k's {topk_diversity:.2f}"
    )

    ########################################################
    # MMR-level — level relevance + shape diversity
    ########################################################
    def mean_level_gap(examples: list[FewShotExample], q_lvl: float) -> float:
        return float(
            np.mean([abs(float(np.mean(e.context_array)) - q_lvl) for e in examples])
        )

    # Properties hold per query; check across all 11 benchmark queries (one
    # query — e.g. iteration_8_ifft — can tie ctx_level when its level-nearest
    # are already shape-diverse, so a single-query > assert is brittle).
    n_anchor = n_div_ge = n_div_gt = n_level = 0
    for split_key, getter in (
        (IN_DISTRIBUTION_ITERATIONS, provider.get_id),
        (OUT_OF_DISTRIBUTION_ITERATIONS, provider.get_ood),
    ):
        for key in split_key:
            qc = getter(key).numpy()[:80]
            q_lvl = float(np.mean(qc))
            level_mmr = select_examples_level_mmr(pool, 5, qc, lambda_=0.5)
            level_nn = select_examples_context_nn(pool, 1, qc, distance="level")[0]
            level_topk = select_examples_context_nn(pool, 5, qc, distance="level")
            mmr_eu = select_examples_mmr(pool, 5, qc)
            n_anchor += level_mmr[-1].trace_id == level_nn.trace_id
            n_div_ge += min_pairwise_dist(level_mmr) >= min_pairwise_dist(level_topk) - 1e-9
            n_div_gt += min_pairwise_dist(level_mmr) > min_pairwise_dist(level_topk) + 1e-9
            n_level += mean_level_gap(level_mmr, q_lvl) < mean_level_gap(mmr_eu, q_lvl)
    # Anchor (most-relevant pick LAST == ctx_level NN) and "never LESS
    # shape-diverse than ctx_level top-k" are exact; strictly-more-diverse and
    # closer-level-than-mmr_euclid hold for the large majority.
    assert n_anchor == 11, f"MMR-level anchor != ctx_level NN for {11 - n_anchor} queries"
    assert n_div_ge == 11, (
        f"MMR-level LESS shape-diverse than ctx_level top-k for {11 - n_div_ge} queries"
    )
    assert n_div_gt >= 8, f"MMR-level only strictly more diverse for {n_div_gt}/11"
    assert n_level >= 10, (
        f"MMR-level only closer-in-level than mmr_euclid for {n_level}/11"
    )
    print(
        f"✓ mmr_level: ctx_level-NN anchor {n_anchor}/11; shape-diversity ≥ "
        f"ctx_level top-k {n_div_ge}/11 (strictly > {n_div_gt}/11); closer in "
        f"level than mmr_euclid {n_level}/11"
    )

    ########################################################
    # End-to-end through the harness (constant ForecastFn)
    ########################################################
    config = FewShotConfig(
        device="cpu",
        model_slug="smoke-test/constant",
        model_prediction_length=64,
        start_context_length=80,
        relevant_prediction_tail=80,
        k_shot=3,
        random_seed=42,
        example_target_length=None,
    )

    def constant_forecast(trace, examples, cfg):
        context = trace[: cfg.start_context_length]
        level = np.full(len(trace) - len(context), context[-1], dtype=trace.dtype)
        return np.concatenate([context, level])

    for strategy in STRATEGIES:
        select_fn = make_select_fn(strategy)
        results = run_benchmark(
            forecast_fn=constant_forecast,
            config=config,
            example_pool=pool,
            method=f"smoke_{strategy}",
            select_fn=select_fn,
            seeds=(42,),
            deterministic=True,
            provider=provider,
            save=False,
        )
        assert results.n_seeds == 1
        assert all(
            len(ids) == 3 for ids in results.per_seed[0].example_ids.values()
        ), f"{strategy}: expected 3 example ids per trace"
    print(f"✓ end-to-end: run_benchmark(deterministic=True, k_shot=3) OK for all {len(STRATEGIES)} strategies")

    print("\n✅ Selection self-tests passed!")
