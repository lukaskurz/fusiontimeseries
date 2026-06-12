"""Training-free operating-parameter conditioning via Chronos-2 covariates (Phase 4).

Conditions Chronos-2 forecasts on the four static operating parameters
(q, shat, rlt, rln) with NO training, through the model's zero-shot covariate
support: ``predict()`` accepts dict tasks ``{"target": 1D, "past_covariates":
{name: 1D, len == target history}, "future_covariates": {name: 1D, len ==
prediction_length}}`` (future keys must be a subset of past keys; NaN is the
library's own missing value, masked anywhere including mid-stream).

THE load-bearing fact (chronos ``chronos_bolt.py`` InstanceNorm, also used by
Chronos-2): every variate row is instance-normalized INDEPENDENTLY with
``loc = nanmean(row)``, ``scale = sqrt(nanmean((row - loc)^2))`` and
``scale = eps=1e-5 where scale == 0``. Consequences, all provable from the
norm being invariant under positive-affine maps ``row -> a*row + b`` (a > 0):

- A CONSTANT covariate row erases its value. In exact arithmetic
  ``(c - c)/eps = 0``; in the library's float32 forward the row mean is off
  by up to 1 ulp, so a constant row normalizes to a constant all-0/±1 row —
  the rounding direction of the mean, a tri-state float32 artifact carrying
  no physics (self-test C5 checks this against the REAL InstanceNorm).
  Different constants give model inputs that are identical up to that
  artifact: passing a static parameter as a constant channel is structurally
  inert — the ``zeroshot+cov`` anchor and the group+cov cells demonstrate
  this empirically.
- Raw parameter values and [0,1]-min-max-normalized values produce identical
  post-norm rows (per-channel positive-affine map), so the choice of encoding
  scale is provably irrelevant; only the WITHIN-ROW relative geometry of
  values survives. Two distinct values over equal supports normalize to
  +/-1 regardless of the values — a k=1 step channel carries exactly one
  sign bit per parameter.
- ``op_knn`` selects examples whose parameters are close to the query's,
  which re-flattens the step channel toward a constant — a STRUCTURAL
  confound between selection-conditioning and covariate-conditioning (the
  analyzer's channel-contrast diagnostic quantifies this).

Static parameters can therefore act ONLY via within-row contrast. Encoding:
**step functions over the flat-concat ICL stream** — example i's normalized
parameters held constant over its [ctx, tgt] segment, the query's parameters
over the (growing) query segment, and ``future_covariates`` = the query's
value repeated over the prediction window. Group mode (Phase 3) has no slot
for per-example parameters: each example is its own covariate row, and a
per-task constant parameter channel is erased by the per-row norm —
``make_chronos2_group_covariate_forecast_fn`` exists to give the "static
covariates are inert in group mode" claim an empirical table row.

The harness is FROZEN and ``ForecastFn(trace, examples, config)`` does not
receive the query's identity, so the forecast fn resolves the query's
parameters BY VALUE: an index over the first 80 context steps of the 11
benchmark traces (``resolve_benchmark_trace_key``), plus a
``query_params_override`` escape hatch for tests and sensitivity checks.

Permuted-params control (``op_covariates="permuted"``): example parameter
values are permuted among the selected examples (query stays true), seeded
``default_rng([crc32(query_key), *example trace_ids])`` — deterministic per
(query, example set), varying across harness seeds through the example set.
Without it, a +cov gain is not attributable to parameter INFORMATION vs the
mere presence of extra covariate rows.

Self-tests: ``uv run python -m
fusiontimeseries.benchmarking.few_shot.covariates`` (CPU, no model
downloads; C1-C10).
"""

import zlib
from functools import cache

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    FewShotExample,
)
from fusiontimeseries.benchmarking.few_shot.harness import ForecastFn
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    OP_NAMES,
    get_params_for_benchmark_trace,
    normalize_params,
)
from fusiontimeseries.benchmarking.few_shot.presentation import _build_group_task
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    IN_DISTRIBUTION_ITERATIONS,
    OUT_OF_DISTRIBUTION_ITERATIONS,
    BenchmarkDataProvider,
    Utils,
)

__all__ = [
    "OP_COVARIATE_MODES",
    "resolve_benchmark_trace_key",
    "normalized_params_or_nan",
    "permute_example_params",
    "build_op_channels",
    "make_chronos2_covariate_forecast_fn",
    "make_chronos2_group_covariate_forecast_fn",
]

OP_COVARIATE_MODES: tuple[str, ...] = ("step", "permuted")


########################################################
# Query-identity resolution (by value; the harness is frozen)
########################################################


@cache
def _benchmark_context_index() -> dict[bytes, str]:
    """Map the first 80 float32 context steps of each benchmark trace -> key."""
    provider = BenchmarkDataProvider()
    index: dict[bytes, str] = {}
    for key in IN_DISTRIBUTION_ITERATIONS:
        index[provider.get_id(key).numpy()[:80].astype(np.float32).tobytes()] = key
    for key in OUT_OF_DISTRIBUTION_ITERATIONS:
        index[provider.get_ood(key).numpy()[:80].astype(np.float32).tobytes()] = key
    assert len(index) == 11, f"Benchmark contexts not unique: {len(index)} of 11"
    return index


def resolve_benchmark_trace_key(trace: NDArray[np.float32]) -> str:
    """Resolve a benchmark trace to its key by its first 80 context values.

    Args:
        trace: Full benchmark trace as passed to a ForecastFn.

    Returns:
        The benchmark trace key (e.g. ``"iteration_8_ifft"``).

    Raises:
        KeyError: If the context matches none of the 11 benchmark traces
            (e.g. a synthetic test trace) — pass ``query_params_override``
            in that case.
    """
    context_bytes = np.ascontiguousarray(trace[:80], dtype=np.float32).tobytes()
    index = _benchmark_context_index()
    if context_bytes not in index:
        raise KeyError(
            "Trace context matches none of the 11 benchmark traces; the "
            "covariate forecast fn resolves query params by value — pass "
            "query_params_override for non-benchmark traces."
        )
    return index[context_bytes]


########################################################
# Channel construction
########################################################


def normalized_params_or_nan(params: dict[str, float] | None) -> dict[str, float]:
    """[0,1]-normalized params, or an all-NaN dict for a params-less example.

    NaN is the library's own missing value (masked by InstanceNorm/attention),
    so the one pool example without a dump match keeps its slot in the stream
    and example sets stay identical to the no-cov twin cells.
    """
    if params is None:
        return {name: float("nan") for name in OP_NAMES}
    return normalize_params(params)


def permute_example_params(
    param_dicts: list[dict[str, float]],
    query_key: str,
    trace_ids: list[int],
) -> list[dict[str, float]]:
    """Permute example param dicts among the selected examples (the control).

    Seeded ``default_rng([crc32(query_key), *trace_ids])``: deterministic per
    (query, example set); different example sets (harness seeds) give
    different permutations. Identity permutations are not excluded — this is
    a distributional control, not a per-cell derangement.
    """
    rng = np.random.default_rng([zlib.crc32(query_key.encode()), *trace_ids])
    return [param_dicts[i] for i in rng.permutation(len(param_dicts))]


def build_op_channels(
    example_segments: list[tuple[dict[str, float], int]],
    query_params: dict[str, float],
    query_length: int,
    future_length: int,
) -> tuple[dict[str, NDArray[np.float32]], dict[str, NDArray[np.float32]]]:
    """Build the 4 step-function channels over the flat-concat ICL stream.

    Per parameter, the past channel holds each example's normalized value
    over its [ctx, tgt] segment followed by the query's value over the
    (growing) query segment; the future channel repeats the query's value
    over the prediction window. Channels are NOT passed through the outer
    scaler — the per-row instance norm is positive-affine-invariant, so any
    such scaling would be erased anyway (module docstring).

    Args:
        example_segments: Per selected example, ``(normalized_params_or_nan
            dict, segment length = len(ctx) + len(tgt))`` in stream order.
        query_params: The query's normalized params (always finite — all 11
            benchmark traces are covered).
        query_length: Current length of the (rollout-extended) query block.
        future_length: Prediction length (the known-future window).

    Returns:
        ``(past_covariates, future_covariates)`` dicts keyed by OP_NAMES;
        past rows have length ``sum(segment lengths) + query_length``.
    """
    past: dict[str, NDArray[np.float32]] = {}
    future: dict[str, NDArray[np.float32]] = {}
    for name in OP_NAMES:
        blocks = [
            np.full(length, params[name], dtype=np.float32)
            for params, length in example_segments
        ]
        blocks.append(np.full(query_length, query_params[name], dtype=np.float32))
        past[name] = np.concatenate(blocks)
        future[name] = np.full(future_length, query_params[name], dtype=np.float32)
    return past, future


########################################################
# Concat presentation + covariates (the Phase-4 forecast fn)
########################################################


def make_chronos2_covariate_forecast_fn(
    pipeline,
    normalization: str = "shared",
    op_covariates: str | None = "step",
    query_params_override: dict[str, float] | None = None,
) -> ForecastFn:
    """Flat-concat ICL rollout through Chronos-2 dict tasks + op channels.

    Clone of ``presentation.make_concat_forecast_fn`` (same query-fit scaler,
    same autoregressive rollout) that predicts through ``pipeline.predict``
    dict tasks instead of a plain tensor, attaching the 4 operating-parameter
    step channels of ``build_op_channels``. ``op_covariates=None`` emits
    ``{"target"}``-only tasks — the path-equivalence control (smoke S2)
    licensing the frozen no-cov cells to run through
    ``make_concat_forecast_fn``.

    Args:
        pipeline: A loaded ``Chronos2Pipeline``
            (``rerun_ksweep.make_chronos2_pipeline``).
        normalization: ``"shared"`` (Phase-3 winner; query-fit scaler on
            examples and query) or ``"per_example"``.
        op_covariates: ``"step"``, ``"permuted"`` (control), or None.
        query_params_override: RAW params dict (q/shat/rlt/rln) to use for
            the query instead of resolving the benchmark trace — for tests
            and sensitivity checks on synthetic traces.

    Returns:
        A harness-compatible ForecastFn.
    """
    if normalization not in ("per_example", "shared"):
        raise ValueError(f"Unknown normalization {normalization!r}")
    if op_covariates is not None and op_covariates not in OP_COVARIATE_MODES:
        raise ValueError(
            f"Unknown op_covariates {op_covariates!r}; expected one of "
            f"{OP_COVARIATE_MODES} or None"
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

        # Operating-param channel ingredients (fixed across the rollout)
        if op_covariates is not None:
            if query_params_override is not None:
                query_norm = normalize_params(query_params_override)
            else:
                query_norm = normalize_params(
                    get_params_for_benchmark_trace(resolve_benchmark_trace_key(trace))
                )
            example_params = [
                normalized_params_or_nan(ex.operating_params) for ex in examples
            ]
            if op_covariates == "permuted" and examples:
                example_params = permute_example_params(
                    example_params,
                    resolve_benchmark_trace_key(trace),
                    [ex.trace_id for ex in examples],
                )
            example_segments = [
                (params, len(ex.context) + len(ex.target))
                for params, ex in zip(example_params, examples)
            ]

        current_query = normed_query_ctx.copy()
        predictions = [initial_query_context]

        # Autoregressive prediction (identical to the frozen harness rollout)
        while len(np.concatenate(predictions)) < trace_length:
            icl_segments = []
            for ex_norm in normalized_examples:
                icl_segments.append(ex_norm["context"])
                icl_segments.append(ex_norm["target"])
            icl_segments.append(current_query)
            icl_context = np.concatenate(icl_segments).astype(np.float32)

            task: dict = {"target": icl_context}
            if op_covariates is not None:
                past, future = build_op_channels(
                    example_segments,
                    query_norm,
                    len(current_query),
                    config.model_prediction_length,
                )
                task["past_covariates"] = past
                task["future_covariates"] = future

            forecast = pipeline.predict(
                inputs=[task],
                prediction_length=config.model_prediction_length,
                cross_learning=False,
            )
            quantiles = forecast[0].permute(0, 2, 1)  # [n_targets=1, pred_len, 9]
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


########################################################
# Group presentation + covariates (the structurally-inert variant)
########################################################


def make_chronos2_group_covariate_forecast_fn(
    pipeline,
    op_covariates: str = "step",
    query_params_override: dict[str, float] | None = None,
) -> ForecastFn:
    """Phase-3 group ICL + 4 constant param channels at the QUERY's params.

    Wraps ``presentation._build_group_task`` (per-example outer norm, exactly
    as Phase 3) and adds one constant covariate row per operating parameter
    (+ the matching future row). Group mode has no slot for per-example
    parameters — each example IS its own covariate row — and a per-task
    CONSTANT channel is erased by Chronos-2's per-row instance norm
    (module docstring). This variant is therefore structurally inert by
    construction; it exists to put that claim in the results table.

    Args:
        pipeline: A loaded ``Chronos2Pipeline``.
        op_covariates: Only ``"step"`` is meaningful here (the channels are
            constants either way; ``"permuted"`` has no per-example slot to
            permute and is rejected).
        query_params_override: RAW params dict for non-benchmark traces.

    Returns:
        A harness-compatible ForecastFn.
    """
    if op_covariates != "step":
        raise ValueError(
            f"Group covariates support only 'step' (got {op_covariates!r}); "
            "there is no per-example slot to permute"
        )

    def forecast_fn(
        trace: NDArray[np.float32],
        examples: list[FewShotExample],
        config: FewShotConfig,
    ) -> NDArray[np.float32]:
        trace_length = trace.shape[0]

        # Per-example outer normalization (EXACT Phase-3 group protocol)
        normalized_rows: list[NDArray[np.float32]] = []
        for ex in examples:
            ex_scaler = StandardScaler()
            normed_ctx = ex_scaler.fit_transform(ex.context_array.reshape(-1, 1)).squeeze()
            normed_tgt = ex_scaler.transform(ex.target_array.reshape(-1, 1)).squeeze()
            normalized_rows.append(
                np.concatenate([normed_ctx, normed_tgt]).astype(np.float32)
            )

        if query_params_override is not None:
            query_norm = normalize_params(query_params_override)
        else:
            query_norm = normalize_params(
                get_params_for_benchmark_trace(resolve_benchmark_trace_key(trace))
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
            common = len(task["target"])
            past = dict(task.get("past_covariates", {}))
            for name in OP_NAMES:
                past[name] = np.full(common, query_norm[name], dtype=np.float32)
            task["past_covariates"] = past
            task["future_covariates"] = {
                name: np.full(
                    config.model_prediction_length, query_norm[name], dtype=np.float32
                )
                for name in OP_NAMES
            }

            forecast = pipeline.predict(
                inputs=[task],
                prediction_length=config.model_prediction_length,
                cross_learning=False,
            )
            quantiles = forecast[0].permute(0, 2, 1)  # [n_targets=1, pred_len, 9]
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
    import torch

    from fusiontimeseries.benchmarking.few_shot.few_shot_utils import create_example_pool
    from fusiontimeseries.benchmarking.few_shot.harness import run_benchmark
    from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
    from fusiontimeseries.benchmarking.few_shot.run_presentation_grid import variant_label
    from fusiontimeseries.benchmarking.few_shot.selection import make_select_fn

    print("Covariates self-tests (Phase 4, CPU, no model downloads)...")

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    assert len(pool) == 245, f"Expected fixed pool of 245, got {len(pool)}"
    provider = BenchmarkDataProvider()
    trace = provider.get_id("iteration_8_ifft").numpy()

    config = FewShotConfig(
        device="cpu",
        model_slug="smoke-test/covariates",
        model_prediction_length=64,
        start_context_length=80,
        relevant_prediction_tail=80,
        k_shot=2,
        random_seed=42,
        example_target_length=None,
    )

    params_a = {"q": 2.0, "shat": 1.0, "rlt": 6.0, "rln": 2.0}
    params_b = {"q": 8.0, "shat": 4.0, "rlt": 11.0, "rln": 6.0}

    ########################################################
    # C1 — step alignment over rollout query lengths
    ########################################################
    segments = [(normalized_params_or_nan(params_a), 267), (normalized_params_or_nan(params_b), 267)]
    query_norm = normalize_params(params_b)
    for qlen in (80, 144, 208, 272):
        past, future = build_op_channels(segments, query_norm, qlen, 64)
        for name in OP_NAMES:
            row = past[name]
            assert len(row) == 534 + qlen, f"C1: row length {len(row)} != {534 + qlen}"
            assert np.allclose(row[:267], segments[0][0][name]), "C1: ex1 block wrong"
            assert np.allclose(row[267:534], segments[1][0][name]), "C1: ex2 block wrong"
            assert np.allclose(row[534:], query_norm[name]), "C1: query block wrong"
    print("✓ C1: step alignment correct at rollout query lengths {80,144,208,272}")

    ########################################################
    # C2 — future channels: 4 keys ⊆ past keys, len 64, constant
    ########################################################
    assert set(future) == set(OP_NAMES) and set(future) <= set(past), (
        "C2: future keys must be the 4 op names and a subset of past keys"
    )
    for name in OP_NAMES:
        assert len(future[name]) == 64 and np.allclose(future[name], query_norm[name]), (
            f"C2: future[{name}] must be the query value x 64"
        )
    print("✓ C2: future covariates = 4 keys ⊆ past, length 64, constant at query value")

    ########################################################
    # C3 — params-less example -> NaN segment, others finite
    ########################################################
    nan_segments = [
        (normalized_params_or_nan(None), 267),
        (normalized_params_or_nan(params_a), 267),
    ]
    past_nan, _ = build_op_channels(nan_segments, query_norm, 80, 64)
    for name in OP_NAMES:
        row = past_nan[name]
        assert np.all(np.isnan(row[:267])), "C3: params-less segment must be NaN"
        assert np.all(np.isfinite(row[267:])), "C3: remaining stream must be finite"
    n_paramless = sum(1 for ex in pool if ex.operating_params is None)
    assert n_paramless == 1, f"C3: expected exactly 1 params-less pool example, got {n_paramless}"
    print("✓ C3: NaN fill for the params-less example; pool has exactly 1 such example")

    ########################################################
    # C4 — k=0 constant channel; op_covariates=None emits target-only
    ########################################################
    past0, future0 = build_op_channels([], query_norm, 80, 64)
    for name in OP_NAMES:
        assert len(past0[name]) == 80 and np.allclose(past0[name], query_norm[name]), (
            "C4: k=0 past channel must be the constant query value over the context"
        )

    seen_tasks: list[dict] = []

    class FakeChronos2Pipeline:
        """Predicts the target's last finite value; validates lengths/keys."""

        def predict(self, inputs, prediction_length, cross_learning=False):
            assert cross_learning is False
            outputs = []
            for task in inputs:
                seen_tasks.append(task)
                target = np.asarray(task["target"], dtype=np.float32)
                for cov_name, row in task.get("past_covariates", {}).items():
                    assert len(row) == len(target), (
                        f"past_covariates[{cov_name}] length {len(row)} != target {len(target)}"
                    )
                future = task.get("future_covariates", {})
                assert set(future) <= set(task.get("past_covariates", {})), (
                    "future keys must be a subset of past keys"
                )
                for cov_name, row in future.items():
                    assert len(row) == prediction_length, (
                        f"future_covariates[{cov_name}] length {len(row)} != {prediction_length}"
                    )
                last = float(target[np.isfinite(target)][-1])
                outputs.append(torch.full((1, 9, prediction_length), last))
            return outputs

    fake = FakeChronos2Pipeline()
    none_fn = make_chronos2_covariate_forecast_fn(fake, op_covariates=None)
    seen_tasks.clear()
    out_none = none_fn(trace, [], config.model_copy(update={"k_shot": 0}))
    assert out_none.shape == trace.shape
    assert all(set(task) == {"target"} for task in seen_tasks), (
        "C4: op_covariates=None must emit target-only tasks"
    )
    print("✓ C4: k=0 channel constant at query value; None mode emits {'target'} only")

    ########################################################
    # C5 — the REAL library per-row instance norm (class import, no download)
    ########################################################
    from chronos.chronos_bolt import InstanceNorm

    def instance_norm_reference(row: NDArray) -> NDArray[np.float32]:
        """The real chronos InstanceNorm forward (float32, per-row)."""
        x = torch.as_tensor(np.asarray(row, dtype=np.float32)).unsqueeze(0)
        out, _ = InstanceNorm()(x)
        return out.squeeze(0).numpy()

    # Raw vs [0,1]-normalized params: identical post-norm (positive-affine map)
    raw_row, _ = build_op_channels(
        [({"q": 2.0, "shat": 1.0, "rlt": 6.0, "rln": 2.0}, 100)],
        {"q": 8.0, "shat": 4.0, "rlt": 11.0, "rln": 6.0},
        100,
        64,
    )
    norm_row, _ = build_op_channels(
        [(normalize_params(params_a), 100)], normalize_params(params_b), 100, 64
    )
    for name in OP_NAMES:
        assert np.allclose(
            instance_norm_reference(raw_row[name]),
            instance_norm_reference(norm_row[name]),
            atol=1e-5,
        ), f"C5: raw vs normalized params must normalize identically ({name})"
    # [2,3] vs [4,6] over equal supports: positive-affine (x2) -> identical ±1
    step_23 = np.concatenate([np.full(100, 2.0), np.full(100, 3.0)])
    step_46 = np.concatenate([np.full(100, 4.0), np.full(100, 6.0)])
    assert np.array_equal(instance_norm_reference(step_23), instance_norm_reference(step_46)), (
        "C5: [2,3] and [4,6] over equal supports must normalize identically"
    )
    assert np.allclose(np.unique(instance_norm_reference(step_23)), [-1.0, 1.0]), (
        "C5: two values over equal supports must normalize to ±1 (one sign bit)"
    )
    # Constant row -> the value is erased up to a float32 tri-state artifact:
    # the row mean rounds within 1 ulp, so the output is a CONSTANT all-0/±1
    # row whose sign is FP-rounding noise, not physics.
    for c in (7.3, 0.5125, 123.456, 0.0):
        out_const = instance_norm_reference(np.full(200, c))
        assert np.ptp(out_const) == 0.0, f"C5: constant row must stay constant (c={c})"
        assert abs(out_const[0]) <= 1.0 + 1e-6, f"C5: constant row out of tri-state (c={c})"
    print(
        "✓ C5: real InstanceNorm — raw≡normalized params, [2,3]≡[4,6] (=±1 sign bit), "
        "constant→all-0/±1 tri-state (value erased)"
    )

    ########################################################
    # C6 — fake pipeline end-to-end through run_benchmark, twin example ids
    ########################################################
    select_fn = make_select_fn("ctx_euclid")
    cov_fn = make_chronos2_covariate_forecast_fn(fake, op_covariates="step")
    for k in (0, 2):
        cfg_k = config.model_copy(
            update={"k_shot": k, "normalization": "shared", "op_covariates": "step"}
        )
        cov_results = run_benchmark(
            forecast_fn=cov_fn,
            config=cfg_k,
            example_pool=pool,
            method="smoke_cov",
            select_fn=select_fn,
            seeds=(42,),
            deterministic=True,
            provider=provider,
            save=False,
        )
        nocov_results = run_benchmark(
            forecast_fn=none_fn,
            config=cfg_k.model_copy(update={"op_covariates": None}),
            example_pool=pool,
            method="smoke_nocov",
            select_fn=select_fn,
            seeds=(42,),
            deterministic=True,
            provider=provider,
            save=False,
        )
        assert all(len(sr.per_trace) == 11 for sr in cov_results.per_seed)
        for sc, sn in zip(cov_results.per_seed, nocov_results.per_seed):
            assert sc.example_ids == sn.example_ids, (
                f"C6: example ids differ between cov and no-cov twins at k={k}"
            )
    print("✓ C6: fake pipeline through run_benchmark k∈{0,2}; twin example ids identical")

    ########################################################
    # C7 — permuted control: deterministic per (query, example set)
    ########################################################
    dicts = [normalized_params_or_nan(p) for p in (params_a, params_b, None)]
    ids_one = [10, 20, 30]
    perm_a = permute_example_params(dicts, "iteration_8_ifft", ids_one)
    perm_b = permute_example_params(dicts, "iteration_8_ifft", ids_one)
    assert perm_a == perm_b, "C7: permutation must be deterministic per (query, ids)"
    variants = {
        tuple(
            tuple(sorted((k, str(v)) for k, v in d.items()))
            for d in permute_example_params(dicts, query, ids)
        )
        for query, ids in (
            ("iteration_8_ifft", ids_one),
            ("iteration_8_ifft", [11, 21, 31]),
            ("iteration_115_ifft", ids_one),
        )
    }
    assert len(variants) > 1, "C7: permutation must vary across (query, example set)"
    # End-to-end: identical channels across repeated calls on the same cell
    perm_fn = make_chronos2_covariate_forecast_fn(fake, op_covariates="permuted")
    examples_k3 = select_fn(pool, 3, 42, trace[:80], "iteration_8_ifft")
    cfg3 = config.model_copy(update={"k_shot": 3})
    seen_tasks.clear()
    perm_fn(trace, examples_k3, cfg3)
    first_channels = {n: seen_tasks[0]["past_covariates"][n].copy() for n in OP_NAMES}
    seen_tasks.clear()
    perm_fn(trace, examples_k3, cfg3)
    for name in OP_NAMES:
        assert np.array_equal(
            first_channels[name], seen_tasks[0]["past_covariates"][name], equal_nan=True
        ), "C7: permuted channels must be identical across repeated calls"
    print("✓ C7: permuted control deterministic per (query, example set), varies across sets")

    ########################################################
    # C8 — resolver: 11 traces resolve; synthetic raises; override bypasses
    ########################################################
    for key in IN_DISTRIBUTION_ITERATIONS:
        assert resolve_benchmark_trace_key(provider.get_id(key).numpy()) == key
    for key in OUT_OF_DISTRIBUTION_ITERATIONS:
        assert resolve_benchmark_trace_key(provider.get_ood(key).numpy()) == key
    rng = np.random.default_rng(0)
    synthetic = (5.0 + rng.normal(size=266)).astype(np.float32)
    try:
        resolve_benchmark_trace_key(synthetic)
        raise AssertionError("C8: synthetic trace must raise KeyError")
    except KeyError:
        pass
    override_fn = make_chronos2_covariate_forecast_fn(
        fake, op_covariates="step", query_params_override=params_a
    )
    out_override = override_fn(synthetic, [], config.model_copy(update={"k_shot": 0}))
    assert out_override.shape == synthetic.shape and np.all(np.isfinite(out_override))
    print("✓ C8: all 11 benchmark traces resolve; synthetic raises; override bypasses")

    ########################################################
    # C9 — 2048-clamp alignment miniature
    ########################################################
    # Chronos-2 left-truncates ALL rows of a task with ONE index range
    # (chronos2/dataset.py): emulate target/channel rows of length 2500
    # sliced to [-2048:] and check the example/query boundary stays aligned.
    long_segments = [(normalize_params(params_a), 1200), (normalize_params(params_b), 1100)]
    past_long, _ = build_op_channels(long_segments, query_norm, 200, 64)
    target_long = np.concatenate(
        [np.zeros(1200), np.ones(1100), np.full(200, 2.0)]
    ).astype(np.float32)
    for name in OP_NAMES:
        assert len(past_long[name]) == len(target_long) == 2500
    clamped_target = target_long[-2048:]
    for name in OP_NAMES:
        clamped_row = past_long[name][-2048:]
        # Boundary ex1->ex2 sits at original index 1200 -> clamped 1200-452=748
        assert np.allclose(clamped_row[:748], normalize_params(params_a)[name])
        assert np.allclose(clamped_row[748:1848], normalize_params(params_b)[name])
        assert np.allclose(clamped_row[1848:], query_norm[name])
    assert np.allclose(clamped_target[747], 0.0) and np.allclose(clamped_target[748], 1.0), (
        "C9: target boundary must sit at the same clamped index as the channel boundary"
    )
    print("✓ C9: one-index-range left clamp keeps target/channel boundaries aligned")

    ########################################################
    # C10 — label/config roundtrip over the v4 grid combos
    ########################################################
    grid_combos = [
        # (presentation, normalization, op_covariates) -> expected label
        (("concat", "shared", None), "shared"),
        (("concat", "shared", "step"), "shared-opcov"),
        (("concat", "shared", "permuted"), "shared-permcov"),
        (("group", "per_example", None), "group"),
        (("group", "per_example", "step"), "group-opcov"),
    ]
    labels = set()
    for (presentation, norm, opcov), expected in grid_combos:
        label = variant_label(presentation, norm, op_covariates=opcov)
        assert label == expected, f"C10: {label!r} != {expected!r}"
        labels.add(label)
        cfg = FewShotConfig(
            device="cpu",
            model_slug="smoke-test/covariates",
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=5,
            presentation=presentation,
            normalization=norm,
            op_covariates=opcov,
        )
        roundtrip = FewShotConfig(**cfg.model_dump())
        assert roundtrip.op_covariates == opcov, "C10: config roundtrip lost op_covariates"
    assert len(labels) == len(grid_combos), "C10: labels must be distinct (parse-invertible)"
    print("✓ C10: v4 variant labels distinct + config roundtrip keeps op_covariates")

    print("\n✅ Covariates self-tests passed!")
