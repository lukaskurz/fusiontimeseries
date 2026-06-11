"""Example presentation formats for few-shot ICL (Phase 3).

Phase 2 found that retrieval does NOT beat random example selection on ID —
and neither does the cheating ``oracle_tail`` for 3/4 models. Diagnosis: the
flat-concat ICL pipeline z-scores every example independently, erasing
absolute LEVEL — the very signal retrieval/oracle matches on (the metric is
tail-LEVEL RMSE). This module tests presentation fixes; ``harness.py`` stays
frozen, everything new lives here:

- ``make_concat_forecast_fn``: the harness rollout parameterized over
  normalization — ``per_example`` (bit-identical to
  ``harness.make_icl_forecast_fn``; self-test T1) vs ``shared`` (ONE
  StandardScaler fit on the query context applied to example contexts+targets
  AND the query, so absolute level survives normalization).
- ``make_chronos2_group_forecast_fn``: Chronos-2 group ICL — examples as
  group rows (``past_covariates`` of a dict task, attended via
  GroupSelfAttention) instead of a spliced concatenation. Fixes the
  splice-discontinuity artifact; does NOT restore level (Chronos-2
  instance-norms each row independently, see the function docstring).
- ``make_ordered_select_fn``: ordering ablation (similar_last = Phase-2
  convention / similar_first / shuffled).
- ``truncate_example`` / ``make_truncated_select_fn``: truncated examples
  (overshoot peak + margin) to fit more examples into fixed context budgets
  (TimesFM max_context=2048, TiRex training length). Truncation is applied
  AFTER selection — oracle_tail ranks by full tails and ctx distances use
  full 80-step contexts; truncating the pool would corrupt rankings and break
  identical-example-set guarantees.

Model-free analogy (``baselines.make_knn_copy_forecast``): ``rescale=False``
copies the neighbours' absolute tail level — the analogue of shared scaling;
``rescale=True`` transfers amplitude through per-context scalers — the
analogue of per-example normalization.

Self-tests: ``uv run python -m
fusiontimeseries.benchmarking.few_shot.presentation`` (CPU, no model
downloads; T1-T7).
"""

import zlib

import numpy as np
from numpy.typing import NDArray
import torch
from sklearn.preprocessing import StandardScaler

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    FewShotExample,
)
from fusiontimeseries.benchmarking.few_shot.harness import (
    ForecastFn,
    PredictFn,
    SelectFn,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import Utils

__all__ = [
    "NORMALIZATIONS",
    "ORDERS",
    "make_concat_forecast_fn",
    "make_ordered_select_fn",
    "truncate_example",
    "make_truncated_select_fn",
    "make_chronos2_group_forecast_fn",
]

NORMALIZATIONS: tuple[str, ...] = ("per_example", "shared")
ORDERS: tuple[str, ...] = ("similar_last", "similar_first", "shuffled")


########################################################
# Concat presentation, parameterized normalization
########################################################


def make_concat_forecast_fn(
    predict_fn: PredictFn, normalization: str = "per_example"
) -> ForecastFn:
    """Flat-concat ICL rollout with selectable example normalization.

    ``per_example`` reproduces ``harness.make_icl_forecast_fn`` bit-for-bit
    (each example z-scored with its own context-fit StandardScaler; self-test
    T1 asserts ``np.array_equal``). ``shared`` instead transforms example
    contexts AND targets with the ONE scaler fit on the query's first
    ``start_context_length`` steps, so an example's absolute level survives
    the normalize -> predict -> denormalize round trip: a model that copies an
    example's tail level denormalizes back to the example's TRUE raw level
    (self-test T2 — the Phase-2 diagnosis in miniature).

    The query-fit scaler (rather than one fit on the concatenation of
    examples + query) keeps the normalization frame independent of the
    selected examples and reduces exactly to zero-shot at k=0.

    Args:
        predict_fn: Model-specific single-step prediction (normalized space).
        normalization: ``"per_example"`` or ``"shared"``.

    Returns:
        A harness-compatible ForecastFn.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {normalization!r}; expected one of {NORMALIZATIONS}"
        )

    def forecast_fn(
        trace: NDArray[np.float32],
        examples: list[FewShotExample],
        config: FewShotConfig,
    ) -> NDArray[np.float32]:
        trace_length = trace.shape[0]

        # Query scaler (fit first so "shared" can reuse it for the examples)
        query_scaler = StandardScaler()
        initial_query_context = trace[: config.start_context_length]
        normed_query_ctx = query_scaler.fit_transform(
            initial_query_context.reshape(-1, 1)
        ).squeeze()

        normalized_examples = []
        for ex in examples:
            if normalization == "per_example":
                ex_scaler = StandardScaler()
                normed_ctx = ex_scaler.fit_transform(ex.context_array.reshape(-1, 1)).squeeze()
                normed_tgt = ex_scaler.transform(ex.target_array.reshape(-1, 1)).squeeze()
            else:  # shared: query-fit scaler, level survives
                normed_ctx = query_scaler.transform(ex.context_array.reshape(-1, 1)).squeeze()
                normed_tgt = query_scaler.transform(ex.target_array.reshape(-1, 1)).squeeze()
            normalized_examples.append({"context": normed_ctx, "target": normed_tgt})

        current_query = normed_query_ctx.copy()
        predictions = [initial_query_context]

        # Autoregressive prediction (identical to the frozen harness rollout)
        while len(np.concatenate(predictions)) < trace_length:
            icl_segments = []
            for ex_norm in normalized_examples:
                icl_segments.append(ex_norm["context"])
                icl_segments.append(ex_norm["target"])
            icl_segments.append(current_query)
            icl_context = np.concatenate(icl_segments)

            median_forecast = np.asarray(
                predict_fn(icl_context.astype(np.float32), config.model_prediction_length)
            ).squeeze()

            denormed_pred = query_scaler.inverse_transform(
                median_forecast.reshape(-1, 1)
            ).squeeze()
            predictions.append(denormed_pred)

            extended_denormed = np.concatenate(predictions)
            current_query = query_scaler.transform(
                extended_denormed.reshape(-1, 1)
            ).squeeze()

        return np.concatenate(predictions)[:trace_length]

    return forecast_fn


########################################################
# Ordering
########################################################


def make_ordered_select_fn(base: SelectFn, order: str) -> SelectFn:
    """Reorder a base SelectFn's picks (which arrive most-similar LAST).

    ``similar_last`` is the identity (the Phase-2 convention: most similar
    example adjacent to the query). ``similar_first`` reverses.
    ``shuffled`` permutes with ``np.random.default_rng([seed,
    crc32(query_key)])`` — deterministic per (seed, query), distinct across
    seeds, so deterministic base strategies (e.g. ctx_euclid) get a real
    seed axis for the ordering ablation (run with ``deterministic=False``).

    Args:
        base: Underlying selection strategy.
        order: One of ``ORDERS``.

    Returns:
        A harness-compatible SelectFn.
    """
    if order not in ORDERS:
        raise ValueError(f"Unknown order {order!r}; expected one of {ORDERS}")

    def select(
        pool: list[FewShotExample],
        k: int,
        seed: int,
        query_context: NDArray[np.float32],
        query_key: str,
    ) -> list[FewShotExample]:
        examples = base(pool, k, seed, query_context, query_key)
        if order == "similar_last":
            return examples
        if order == "similar_first":
            return list(reversed(examples))
        rng = np.random.default_rng([seed, zlib.crc32(query_key.encode())])
        return [examples[i] for i in rng.permutation(len(examples))]

    return select


########################################################
# Truncation
########################################################


def truncate_example(
    ex: FewShotExample,
    margin: int = 64,
    min_length: int = 96,
    peak_window: int = 80,
) -> FewShotExample:
    """Truncate an example shortly after its overshoot peak.

    The linear growth phase ends at the overshoot peak (within the 80-step
    context window for all valid traces); ``margin`` steps after it cover the
    transition into saturation. ``end = clip(argmax(trace[:peak_window]) + 1
    + margin, min_length, len(trace))``; the context (first 80 steps) is
    always preserved since ``min_length >= 96 > 80``.

    Args:
        ex: Source example (ids and operating params are carried over).
        margin: Steps kept after the peak.
        min_length: Floor on the truncated length (early-peak traces).
        peak_window: Window for the peak search (the context length).

    Returns:
        A new FewShotExample whose trace is a prefix of the original.
    """
    peak = int(np.argmax(ex.trace_array[:peak_window]))
    end = int(np.clip(peak + 1 + margin, min_length, len(ex.trace)))
    context_length = len(ex.context)
    # Slice the stored lists (not trace_array) so values stay bit-identical
    return FewShotExample(
        trace_id=ex.trace_id,
        pool_index=ex.pool_index,
        operating_params=ex.operating_params,
        trace=ex.trace[:end],
        context=ex.trace[:context_length],
        target=ex.trace[context_length:end],
    )


def make_truncated_select_fn(base: SelectFn, margin: int = 64) -> SelectFn:
    """Apply ``truncate_example`` AFTER selection (never to the pool).

    Selection must see full traces: oracle_tail ranks by full tails and
    context distances use the full 80-step contexts; truncating the pool
    would corrupt rankings and break identical-example-set comparisons.

    Args:
        base: Underlying selection strategy.
        margin: Steps kept after the overshoot peak.

    Returns:
        A harness-compatible SelectFn yielding truncated examples.
    """

    def select(
        pool: list[FewShotExample],
        k: int,
        seed: int,
        query_context: NDArray[np.float32],
        query_key: str,
    ) -> list[FewShotExample]:
        return [
            truncate_example(ex, margin=margin)
            for ex in base(pool, k, seed, query_context, query_key)
        ]

    return select


########################################################
# Chronos-2 group ICL
########################################################


def _build_group_task(
    normalized_example_rows: list[NDArray[np.float32]],
    normalized_query: NDArray[np.float32],
) -> dict:
    """Build one Chronos-2 dict task: query target + example covariate rows.

    Rows within a task must have equal length (the library validates), so all
    rows are LEFT-padded with NaN — NaN is the library's own padding value,
    handled by masking — to the common (max) length. Covers both directions:
    a short query against full 267-step examples and a rollout-extended query
    longer than the examples. k=0 omits ``past_covariates`` entirely.

    Args:
        normalized_example_rows: Normalized ``[ctx, tgt]`` example rows.
        normalized_query: Normalized (growing) query context.

    Returns:
        ``{"target": ...}`` or ``{"target": ..., "past_covariates": {...}}``.
    """
    query = np.asarray(normalized_query, dtype=np.float32)
    if not normalized_example_rows:
        return {"target": query}
    rows = [np.asarray(row, dtype=np.float32) for row in normalized_example_rows]
    common = max(len(query), max(len(row) for row in rows))

    def left_pad(x: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(x) == common:
            return x
        pad = np.full(common - len(x), np.nan, dtype=np.float32)
        return np.concatenate([pad, x])

    covariates = {f"ex_{i:02d}": left_pad(row) for i, row in enumerate(rows)}
    return {"target": left_pad(query), "past_covariates": covariates}


def make_chronos2_group_forecast_fn(pipeline) -> ForecastFn:
    """Chronos-2 group ICL: examples as group rows instead of spliced concat.

    EXACT Phase-1 normalization and 64-step autoregressive rollout — only the
    presentation changes: per predict call the model sees ONE dict task whose
    ``past_covariates`` are the k per-example-normalized ``[ctx, tgt]`` rows
    (267 steps) and whose ``target`` is the growing normalized query. Within
    a task, rows attend to each other (GroupSelfAttention) and only the
    target is forecast; ``cross_learning=False`` keeps benchmark queries
    independent. Median via ``Utils.median_forecast`` after permuting the
    ``(n_variates, n_quantiles, pred_len)`` output.

    NOTE on normalization: Chronos-2 instance-norms each row internally and
    independently, so OUR outer scalers are mathematically inert for it —
    group ICL does NOT restore absolute level. It fixes the
    splice-discontinuity artifact; shared scaling addresses level. The outer
    per-example scalers are kept anyway for protocol identity with the
    concat pipeline (identical inputs up to presentation).

    Args:
        pipeline: A loaded ``Chronos2Pipeline``
            (``rerun_ksweep.make_chronos2_pipeline``).

    Returns:
        A harness-compatible ForecastFn.
    """

    def forecast_fn(
        trace: NDArray[np.float32],
        examples: list[FewShotExample],
        config: FewShotConfig,
    ) -> NDArray[np.float32]:
        trace_length = trace.shape[0]

        # Normalize examples independently (EXACT Phase-1 normalization)
        normalized_rows: list[NDArray[np.float32]] = []
        for ex in examples:
            ex_scaler = StandardScaler()
            normed_ctx = ex_scaler.fit_transform(ex.context_array.reshape(-1, 1)).squeeze()
            normed_tgt = ex_scaler.transform(ex.target_array.reshape(-1, 1)).squeeze()
            normalized_rows.append(
                np.concatenate([normed_ctx, normed_tgt]).astype(np.float32)
            )

        query_scaler = StandardScaler()
        initial_query_context = trace[: config.start_context_length]
        normed_query_ctx = query_scaler.fit_transform(
            initial_query_context.reshape(-1, 1)
        ).squeeze()

        current_query = normed_query_ctx.copy()
        predictions = [initial_query_context]

        while len(np.concatenate(predictions)) < trace_length:
            task = _build_group_task(normalized_rows, current_query.astype(np.float32))
            forecast = pipeline.predict(
                inputs=[task],
                prediction_length=config.model_prediction_length,
                cross_learning=False,
            )
            quantiles = forecast[0].permute(0, 2, 1)  # [n_variates=1, pred_len, 9]
            median_forecast = Utils.median_forecast(quantiles).squeeze().cpu().numpy()

            denormed_pred = query_scaler.inverse_transform(
                median_forecast.reshape(-1, 1)
            ).squeeze()
            predictions.append(denormed_pred)

            extended_denormed = np.concatenate(predictions)
            current_query = query_scaler.transform(
                extended_denormed.reshape(-1, 1)
            ).squeeze()

        return np.concatenate(predictions)[:trace_length]

    return forecast_fn


if __name__ == "__main__":
    from fusiontimeseries.benchmarking.few_shot.few_shot_utils import create_example_pool
    from fusiontimeseries.benchmarking.few_shot.harness import (
        make_icl_forecast_fn,
        run_benchmark,
    )
    from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
    from fusiontimeseries.benchmarking.few_shot.selection import make_select_fn
    from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
        IN_DISTRIBUTION_ITERATIONS,
        OUT_OF_DISTRIBUTION_ITERATIONS,
        BenchmarkDataProvider,
    )

    print("Presentation self-tests (Phase 3, CPU, no model downloads)...")

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    assert len(pool) == 245, f"Expected fixed pool of 245, got {len(pool)}"
    provider = BenchmarkDataProvider()
    trace = provider.get_id("iteration_8_ifft").numpy()

    config = FewShotConfig(
        device="cpu",
        model_slug="smoke-test/presentation",
        model_prediction_length=64,
        start_context_length=80,
        relevant_prediction_tail=80,
        k_shot=3,
        random_seed=42,
        example_target_length=None,
    )

    ########################################################
    # T1 — per_example is bit-identical to the frozen harness
    ########################################################
    def last_value_predict(
        context: NDArray[np.float32], prediction_length: int
    ) -> NDArray[np.float32]:
        return np.full(prediction_length, context[-1], dtype=np.float32)

    examples = make_select_fn("ctx_euclid")(pool, 3, 42, trace[:80], "iteration_8_ifft")
    harness_out = make_icl_forecast_fn(last_value_predict)(trace, examples, config)
    ours_out = make_concat_forecast_fn(last_value_predict, "per_example")(
        trace, examples, config
    )
    assert np.array_equal(harness_out, ours_out), (
        "T1: per_example concat must be bit-identical to harness.make_icl_forecast_fn"
    )
    print("✓ T1: per_example output bit-identical to the frozen harness rollout")

    ########################################################
    # T2 — shared scaling preserves absolute level; per_example does not
    ########################################################
    rng = np.random.default_rng(0)
    example_tail_level = 37.5
    ex_ctx = 10.0 + rng.normal(0.0, 2.0, size=80)
    ex_tgt = np.full(187, example_tail_level)
    synth_example = FewShotExample(
        trace_id=-1,
        pool_index=-1,
        operating_params=None,
        trace=np.concatenate([ex_ctx, ex_tgt]).tolist(),
        context=ex_ctx.tolist(),
        target=ex_tgt.tolist(),
    )
    # Query on a different scale entirely
    query_trace = (5.0 + rng.normal(0.0, 1.0, size=266)).astype(np.float32)
    # PredictFn that "copies the example's tail level": with k=1 the concat is
    # [ex_ctx(80), ex_tgt(187), query]; index 266 is the example's last tail
    # value in normalized space.
    ex_tail_index = 80 + 187 - 1

    def copy_example_tail_predict(
        context: NDArray[np.float32], prediction_length: int
    ) -> NDArray[np.float32]:
        return np.full(prediction_length, context[ex_tail_index], dtype=np.float32)

    config_k1 = config.model_copy(update={"k_shot": 1})
    shared_fc = make_concat_forecast_fn(copy_example_tail_predict, "shared")(
        query_trace, [synth_example], config_k1
    )
    per_ex_fc = make_concat_forecast_fn(copy_example_tail_predict, "per_example")(
        query_trace, [synth_example], config_k1
    )
    assert np.allclose(shared_fc[80:], example_tail_level, rtol=1e-6), (
        f"T2: shared scaling must round-trip the example tail level "
        f"{example_tail_level}, got {shared_fc[-1]:.4f}"
    )
    assert not np.allclose(per_ex_fc[80:], example_tail_level, rtol=0.05), (
        f"T2: per_example must NOT recover the absolute level, got {per_ex_fc[-1]:.4f}"
    )
    print(
        f"✓ T2: level {example_tail_level} survives shared scaling exactly "
        f"({shared_fc[-1]:.4f}); per_example maps it to {per_ex_fc[-1]:.4f}"
    )

    ########################################################
    # T3 — shared-normalized value range over pool x query scalers
    ########################################################
    query_scalers = []
    for key in IN_DISTRIBUTION_ITERATIONS:
        query_scalers.append(StandardScaler().fit(provider.get_id(key).numpy()[:80].reshape(-1, 1)))
    for key in OUT_OF_DISTRIBUTION_ITERATIONS:
        query_scalers.append(StandardScaler().fit(provider.get_ood(key).numpy()[:80].reshape(-1, 1)))
    abs_values = []
    for scaler in query_scalers:
        for ex in pool:
            normed = scaler.transform(ex.trace_array.reshape(-1, 1)).squeeze()
            assert np.all(np.isfinite(normed)), "T3: non-finite shared-normalized value"
            abs_values.append(np.max(np.abs(normed)))
    abs_values = np.array(abs_values)
    print(
        f"✓ T3: shared-normalized |values| finite over {len(pool)} examples x "
        f"{len(query_scalers)} query scalers; per-example max |z|: "
        f"p50={np.percentile(abs_values, 50):.1f}, p99={np.percentile(abs_values, 99):.1f}, "
        f"max={abs_values.max():.1f}"
    )

    ########################################################
    # T4 — ordering: identity / reversal / deterministic shuffle
    ########################################################
    base = make_select_fn("ctx_euclid")
    picks = base(pool, 10, 0, trace[:80], "iteration_8_ifft")
    ids = [ex.trace_id for ex in picks]
    same = make_ordered_select_fn(base, "similar_last")(pool, 10, 0, trace[:80], "iteration_8_ifft")
    assert [ex.trace_id for ex in same] == ids, "T4: similar_last must be identity"
    rev = make_ordered_select_fn(base, "similar_first")(pool, 10, 0, trace[:80], "iteration_8_ifft")
    assert [ex.trace_id for ex in rev] == ids[::-1], "T4: similar_first must reverse"
    shuffle_fn = make_ordered_select_fn(base, "shuffled")
    shuf_a = [ex.trace_id for ex in shuffle_fn(pool, 10, 0, trace[:80], "iteration_8_ifft")]
    shuf_b = [ex.trace_id for ex in shuffle_fn(pool, 10, 0, trace[:80], "iteration_8_ifft")]
    shuf_seed1 = [ex.trace_id for ex in shuffle_fn(pool, 10, 1, trace[:80], "iteration_8_ifft")]
    assert shuf_a == shuf_b, "T4: shuffled must be deterministic per (seed, query_key)"
    assert sorted(shuf_a) == sorted(ids), "T4: shuffled must be a permutation"
    assert shuf_a != shuf_seed1, "T4: shuffled must differ across seeds"
    print("✓ T4: similar_last identity, similar_first reversal, shuffled deterministic per (seed, query) and seed-distinct")

    ########################################################
    # T5 — truncation: prefix property, bounds, context intact
    ########################################################
    full_len = len(pool[0].trace)
    lengths = []
    for ex in pool:
        tr = truncate_example(ex, margin=64)
        assert np.array_equal(tr.trace_array, ex.trace_array[: len(tr.trace)]), (
            "T5: truncated trace must be a prefix"
        )
        assert 96 <= len(tr.trace) <= full_len, f"T5: length {len(tr.trace)} out of bounds"
        assert tr.context == ex.context, "T5: 80-step context must be intact"
        peak = int(np.argmax(ex.trace_array[:80]))
        assert peak < len(tr.trace), "T5: peak must be inside the truncated window"
        assert (tr.trace_id, tr.pool_index) == (ex.trace_id, ex.pool_index)
        lengths.append(len(tr.trace))
    lengths = np.array(lengths)
    print(
        f"✓ T5: truncation (margin=64) over {len(pool)} examples: lengths "
        f"min/mean/max = {lengths.min()}/{lengths.mean():.0f}/{lengths.max()} "
        f"(full {full_len})"
    )
    for k in (5, 10, 20):
        print(
            f"    TimesFM-2048 budget @ k={k}: truncated ~{k * lengths.mean() + 80:.0f} "
            f"steps vs full {k * full_len + 80} (limit 2048)"
        )

    ########################################################
    # T6 — group-task builder: shapes and NaN layout
    ########################################################
    rows_267 = [np.ones(267, dtype=np.float32) * (i + 1) for i in range(2)]
    for qlen in (80, 144, 208, 272):
        query = np.zeros(qlen, dtype=np.float32)
        task0 = _build_group_task([], query)
        assert set(task0) == {"target"} and len(task0["target"]) == qlen, (
            "T6: k=0 must omit past_covariates and not pad"
        )
        task2 = _build_group_task(rows_267, query)
        common = max(qlen, 267)
        assert len(task2["target"]) == common
        assert list(task2["past_covariates"]) == ["ex_00", "ex_01"]
        n_query_nan = int(np.isnan(task2["target"]).sum())
        assert n_query_nan == common - qlen, "T6: query must be left-NaN-padded"
        assert np.all(np.isfinite(task2["target"][common - qlen :]))
        for row in task2["past_covariates"].values():
            n_row_nan = int(np.isnan(row).sum())
            assert n_row_nan == common - 267, "T6: example rows must be left-NaN-padded"
            assert np.all(np.isfinite(row[common - 267 :]))
    print("✓ T6: group tasks for k in {0,2} x query lengths {80,144,208,272}: shapes + left-NaN layout OK")

    ########################################################
    # T7 — mock Chronos-2 pipeline end-to-end through run_benchmark
    ########################################################
    class FakeChronos2Pipeline:
        """Predicts the target's last finite value; validates row lengths."""

        def predict(self, inputs, prediction_length, cross_learning=False):
            assert cross_learning is False
            outputs = []
            for task in inputs:
                target = np.asarray(task["target"], dtype=np.float32)
                for row in task.get("past_covariates", {}).values():
                    assert len(row) == len(target), "rows within a task must match"
                last = float(target[np.isfinite(target)][-1])
                outputs.append(torch.full((1, 9, prediction_length), last))
            return outputs

    group_fn = make_chronos2_group_forecast_fn(FakeChronos2Pipeline())
    mock_results = run_benchmark(
        forecast_fn=group_fn,
        config=config.model_copy(update={"k_shot": 2, "presentation": "group"}),
        example_pool=pool,
        method="smoke_group_mock",
        seeds=(0,),
        provider=provider,
        save=False,
    )
    assert mock_results.n_seeds == 1
    assert all(len(sr.per_trace) == 11 for sr in mock_results.per_seed)
    single = group_fn(trace, [pool[0], pool[1]], config.model_copy(update={"k_shot": 2}))
    assert single.shape == trace.shape
    assert np.allclose(single[:80], trace[:80])
    print(
        f"✓ T7: mock group pipeline through run_benchmark "
        f"(ID {mock_results.in_distribution.rmse:.2f}, "
        f"OOD {mock_results.out_of_distribution.rmse:.2f}); rollout shapes OK"
    )

    print("\n✅ Presentation self-tests passed!")
