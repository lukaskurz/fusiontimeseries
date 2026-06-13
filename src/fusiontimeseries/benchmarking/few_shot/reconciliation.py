"""Evaluation reconciliation: re-run the ladder on Severin's ``[0::3]`` phase (Phase 8).

The two halves of the project evaluate on traces subsampled from the SAME 800-step
raw simulations but at a different phase:

- OUR few-shot side scores the ``[2::3]`` subsample (266 steps), built by
  ``BenchmarkDataProvider`` from ``data/flux/benchmark/flux_data.json``
  (``benchmarking/zero_shot/benchmark_utils.py``).
- Severin's finetuning side scores the ``[0::3]`` subsample (267 steps), built by
  ``Chronos2Dataset.get_benchmark_flux_traces`` with stride = window = 3
  (``lib/dataset.py``; the path ``finetuned.severin_anchor_eval`` already uses).

Both pull the SAME raw ids (ID 8/115/131/148/235/262; OOD 0–4) — only the
subsample phase differs. The saturation level the benchmark metric reads (mean of
the last 80 steps) is a property of the simulation, not the phase, so the two
should agree up to a small per-trace delta. This module makes that empirical:

- ``Phase0BenchmarkProvider`` is duck-compatible with ``BenchmarkDataProvider``
  (``get_id``/``get_ood`` -> 267-step ``[0::3]`` tensors) keyed by the exact
  ``IN_/OUT_OF_DISTRIBUTION_ITERATIONS`` trace keys via the operating-params
  mapping (ID -> raw_id, OOD -> ``int(dump_key) - 4000``, the same routing
  ``run_finetuned_grid.smoke`` F2b uses). Drop it into ``run_benchmark(provider=)``
  and every ladder cell re-runs on Severin's phase.
- ``phase0_tail_mean_deltas`` reports the per-trace ``[0::3]``-vs-``[2::3]``
  true-tail-mean delta — the comparability evidence.
- ``make_phase0_finetuned_forecast_fn`` is the finetuned ICL forecast fn for the
  ``[0::3]`` traces. ``finetuned.make_finetuned_forecast_fn`` resolves the query's
  operating params through ``covariates.resolve_benchmark_trace_key``, whose
  by-value index is built from the ``[2::3]`` provider — so it raises on a
  ``[0::3]`` context. This builds an equivalent index over the Phase-0 provider's
  ``[0::3]`` contexts and is otherwise protocol-identical (CPU op_params,
  ``ConditionRegistry.patch`` per predict, ``make_concat_forecast_fn`` rollout).

``oracle_tail`` is deliberately NOT a ladder rung here: ``selection`` ranks it
against ``_benchmark_true_tail_means``, cached against the ``[2::3]`` provider —
re-keying it to ``[0::3]`` is out of scope (and the ladder rungs are all
phase-correct: mmr/ctx use context distance, baselines use pool tails).

Self-test (CPU, no model downloads):
    uv run python -m fusiontimeseries.benchmarking.few_shot.reconciliation
"""

import numpy as np
from numpy.typing import NDArray
import torch

from fusiontimeseries.benchmarking.few_shot.finetuned import (
    raw_param_tensor,
)
from fusiontimeseries.benchmarking.few_shot.harness import ForecastFn
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    ID_TEST_RAW_IDS,
    get_params_for_benchmark_trace,
    load_mapping,
)
from fusiontimeseries.benchmarking.few_shot.presentation import (
    make_concat_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    chronos2_predict_from_pipeline,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    IN_DISTRIBUTION_ITERATIONS,
    OUT_OF_DISTRIBUTION_ITERATIONS,
    BenchmarkDataProvider,
)
from fusiontimeseries.lib.conditioning import ConditionRegistry

__all__ = [
    "PHASE0_STEPS",
    "Phase0BenchmarkProvider",
    "phase0_context_index",
    "phase0_tail_mean_deltas",
    "make_phase0_finetuned_forecast_fn",
]

#: Length of a ``[0::3]`` subsample of an 800-step raw trace (ceil(800/3)).
PHASE0_STEPS: int = 267


class Phase0BenchmarkProvider:
    """Benchmark provider over Severin's ``[0::3]`` subsample (267 steps).

    Duck-compatible with ``BenchmarkDataProvider``: the same ``get_id(key)`` /
    ``get_ood(key)`` API returning ``torch.Tensor``, keyed by the same
    ``IN_/OUT_OF_DISTRIBUTION_ITERATIONS`` trace keys — so it drops straight into
    ``harness.run_benchmark(provider=...)``. The underlying traces come from
    ``Chronos2Dataset.get_benchmark_flux_traces`` (stride = window = 3 ->
    ``energy_flux[0::3]``), the exact path ``severin_anchor_eval`` and the
    finetuning notebooks evaluate on.

    The flux ids are routed exactly as ``run_finetuned_grid.smoke`` (F2b): ID
    trace -> ``raw_id``, OOD trace -> ``int(dump_key) - 4000``.
    """

    def __init__(self, device: str = "cpu") -> None:
        from fusiontimeseries.finetuning.chronos2.dataset import Chronos2Dataset
        from fusiontimeseries.finetuning.chronos2.train_bilinear import (
            build_fts_config,
            ensure_flat_flux_data,
        )

        fts_config = build_fts_config(device, max_steps=4000)
        fts_config.data_path = ensure_flat_flux_data()
        benchmark_data = Chronos2Dataset.get_benchmark_flux_traces(fts_config)
        mapping = load_mapping()["benchmark_traces"]

        self._id: dict[str, torch.Tensor] = {}
        self._ood: dict[str, torch.Tensor] = {}
        for trace_key, entry in mapping.items():
            if entry["kind"] == "id":
                flux_data = benchmark_data["id"][entry["raw_id"]]
                target = self._id
            else:
                flux_data = benchmark_data["ood"][int(entry["dump_key"]) - 4000]
                target = self._ood
            target[trace_key] = torch.tensor(
                np.asarray(flux_data.energy_flux, dtype=np.float32), dtype=torch.float32
            )

        assert set(self._id) == set(IN_DISTRIBUTION_ITERATIONS), (
            f"Phase-0 ID keys {sorted(self._id)} != benchmark ID keys"
        )
        assert set(self._ood) == set(OUT_OF_DISTRIBUTION_ITERATIONS), (
            f"Phase-0 OOD keys {sorted(self._ood)} != benchmark OOD keys"
        )
        for key, tensor in {**self._id, **self._ood}.items():
            assert tensor.shape == (PHASE0_STEPS,), (
                f"Phase-0 trace {key} has {tuple(tensor.shape)} steps, "
                f"expected ({PHASE0_STEPS},)"
            )

    def get_id(self, iteration: str) -> torch.Tensor:
        """Get the ``[0::3]`` in-distribution trace for a benchmark key."""
        return self._id[iteration]

    def get_ood(self, iteration: str) -> torch.Tensor:
        """Get the ``[0::3]`` out-of-distribution trace for a benchmark key."""
        return self._ood[iteration]

    def items(self):
        """Yield ``(trace_key, tensor)`` for all 11 benchmark traces."""
        yield from {**self._id, **self._ood}.items()


def phase0_context_index(provider: Phase0BenchmarkProvider) -> dict[bytes, str]:
    """Map the first 80 ``[0::3]`` context steps of each trace -> its key.

    The by-value analogue of ``covariates._benchmark_context_index`` for the
    ``[0::3]`` phase, so a finetuned forecast fn can resolve the query's
    operating params without the (frozen) harness passing the trace key.
    """
    index: dict[bytes, str] = {}
    for key, tensor in provider.items():
        ctx = np.ascontiguousarray(tensor.numpy()[:80], dtype=np.float32).tobytes()
        index[ctx] = key
    assert len(index) == 11, f"Phase-0 contexts not unique: {len(index)} of 11"
    return index


def phase0_tail_mean_deltas(
    phase0: Phase0BenchmarkProvider | None = None,
    benchmark: BenchmarkDataProvider | None = None,
    tail: int = 80,
) -> dict[str, dict[str, float]]:
    """Per-trace ``[0::3]``-vs-``[2::3]`` true-tail-mean comparison.

    The empirical comparability evidence: the benchmark metric reads the mean of
    the last ``tail`` steps, which should be ~phase-invariant (it is the
    saturation level of the same simulation). Reports the two tail means and
    their absolute / relative delta for each of the 11 benchmark traces.

    Returns:
        ``{trace_key: {"phase0": .., "phase2": .., "abs_delta": .., "rel_delta": ..}}``.
    """
    phase0 = phase0 or Phase0BenchmarkProvider()
    benchmark = benchmark or BenchmarkDataProvider()
    out: dict[str, dict[str, float]] = {}
    for keys, getter0, getter2 in (
        (IN_DISTRIBUTION_ITERATIONS, phase0.get_id, benchmark.get_id),
        (OUT_OF_DISTRIBUTION_ITERATIONS, phase0.get_ood, benchmark.get_ood),
    ):
        for key in keys:
            m0 = float(np.mean(getter0(key).numpy()[-tail:]))
            m2 = float(np.mean(getter2(key).numpy()[-tail:]))
            out[key] = {
                "phase0": m0,
                "phase2": m2,
                "abs_delta": abs(m0 - m2),
                "rel_delta": abs(m0 - m2) / max(1.0, abs(m2)),
            }
    return out


def make_phase0_finetuned_forecast_fn(
    pipeline,
    provider: Phase0BenchmarkProvider,
    point_stat: str = "mean",
    normalization: str = "shared",
) -> ForecastFn:
    """Finetuned ICL forecast fn for the ``[0::3]`` traces.

    Protocol-identical to ``finetuned.make_finetuned_forecast_fn`` (CPU
    op_params, ``ConditionRegistry.patch`` around each ``predict``, the frozen
    ``make_concat_forecast_fn`` rollout) — only the query-identity resolver
    differs: it indexes the Phase-0 provider's ``[0::3]`` contexts instead of the
    ``[2::3]`` benchmark contexts (whose 80-step prefixes do not match). The
    operating params themselves are phase-independent (one simulation, both
    phases), so the conditioning tensor is identical to the ``[2::3]`` twin cell.

    Args:
        pipeline: A ``Chronos2Pipeline`` from ``finetuned.load_finetuned_chronos2``.
        provider: The Phase-0 provider whose traces will be forecast.
        point_stat: ``"mean"`` (the v5 Chronos-2 default) or ``"median"``.
        normalization: ``"shared"`` (Phase-3 winner) or ``"per_example"``.

    Returns:
        A harness-compatible ForecastFn.
    """
    base_predict = chronos2_predict_from_pipeline(pipeline, point_stat)
    index = phase0_context_index(provider)

    def forecast_fn(trace, examples, config) -> NDArray[np.float32]:
        ctx_bytes = np.ascontiguousarray(trace[:80], dtype=np.float32).tobytes()
        if ctx_bytes not in index:
            raise KeyError(
                "Trace context matches none of the 11 Phase-0 benchmark traces; "
                "make_phase0_finetuned_forecast_fn resolves query params by value."
            )
        op_params = raw_param_tensor(get_params_for_benchmark_trace(index[ctx_bytes]))

        def conditioned_predict(
            context: NDArray[np.float32], prediction_length: int
        ) -> NDArray[np.float32]:
            with ConditionRegistry.patch(op_params=op_params):
                return base_predict(context, prediction_length)

        return make_concat_forecast_fn(conditioned_predict, normalization)(
            trace, examples, config
        )

    return forecast_fn


if __name__ == "__main__":
    from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
        FewShotConfig,
        create_example_pool,
    )
    from fusiontimeseries.benchmarking.few_shot.finetuned import FINETUNED_SLUG
    from fusiontimeseries.benchmarking.few_shot.harness import run_benchmark
    from fusiontimeseries.loralib.layers import OP_PARAM_KEY

    print("Reconciliation self-tests (Phase 8, CPU, no model downloads)...")

    # R1 — provider: 11 keys, 267 steps each, duck-compatible getters
    phase0 = Phase0BenchmarkProvider()
    benchmark = BenchmarkDataProvider()
    n_id = sum(1 for _ in IN_DISTRIBUTION_ITERATIONS)
    for key in IN_DISTRIBUTION_ITERATIONS:
        t0 = phase0.get_id(key)
        assert t0.shape == (PHASE0_STEPS,) and torch.all(torch.isfinite(t0))
    for key in OUT_OF_DISTRIBUTION_ITERATIONS:
        t0 = phase0.get_ood(key)
        assert t0.shape == (PHASE0_STEPS,) and torch.all(torch.isfinite(t0))
    assert len(list(phase0.items())) == 11
    # The [0::3] trace must differ from the [2::3] trace (different phase)
    sample = phase0.get_id(IN_DISTRIBUTION_ITERATIONS[0]).numpy()
    sample2 = benchmark.get_id(IN_DISTRIBUTION_ITERATIONS[0]).numpy()
    assert sample.shape == (267,) and sample2.shape == (266,)
    assert not np.array_equal(sample[:80], sample2[:80]), (
        "R1: [0::3] and [2::3] contexts must differ (different subsample phase)"
    )
    print(f"✓ R1: Phase0 provider — 11 keys, 267 steps, [0::3] context ≠ [2::3] context")

    # R2 — leakage guard: none of the 6 ID test raw ids in the fixed pool
    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    assert len(pool) == 245, f"Expected fixed pool of 245, got {len(pool)}"
    leaked = sorted({ex.trace_id for ex in pool} & ID_TEST_RAW_IDS)
    assert not leaked, f"R2: ID test raw ids leaked into the pool: {leaked}"
    print(
        f"✓ R2: leakage guard — none of {sorted(ID_TEST_RAW_IDS)} in the 245-trace "
        f"pool (the [0::3] ID queries are the pool-excluded twins)"
    )

    # R3 — phase-invariance: per-trace [0::3]-vs-[2::3] true-tail-mean deltas
    deltas = phase0_tail_mean_deltas(phase0, benchmark)
    assert set(deltas) == set(IN_DISTRIBUTION_ITERATIONS) | set(OUT_OF_DISTRIBUTION_ITERATIONS)
    rels = np.array([d["rel_delta"] for d in deltas.values()])
    abss = np.array([d["abs_delta"] for d in deltas.values()])
    print("✓ R3: [0::3]-vs-[2::3] true-tail-mean deltas (phase invariance evidence):")
    for key, d in deltas.items():
        print(
            f"    {key:38s} [0::3] {d['phase0']:8.3f}  [2::3] {d['phase2']:8.3f}  "
            f"|Δ| {d['abs_delta']:7.4f}  rel {d['rel_delta']:.4f}"
        )
    print(
        f"    -> rel delta: median {np.median(rels):.4f}, max {rels.max():.4f}; "
        f"abs delta: median {np.median(abss):.4f}, max {abss.max():.4f}"
    )

    # R4 — fake pipeline through run_benchmark: param resolution + conditioning
    class _FakeChronos2Pipeline:
        """chronos2 layout [1, n_q, pred_len]; requires the patch; records params."""

        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def predict(self, inputs: torch.Tensor, prediction_length: int):
            p = ConditionRegistry.get(OP_PARAM_KEY)
            assert p is not None, "predict called outside ConditionRegistry.patch"
            self.calls.append(p.clone())
            last = float(inputs[0, 0, -1])
            base = torch.full((1, 9, prediction_length), last)
            offsets = torch.linspace(-0.5, 0.5, 9).view(1, 9, 1)
            return [base + offsets]

    fake = _FakeChronos2Pipeline()
    fn = make_phase0_finetuned_forecast_fn(fake, phase0, "mean", "shared")
    # Per-trace conditioning must match the trace's own params, resolved from
    # the [0::3] context (NOT the [2::3] index).
    for getter, key in (
        (phase0.get_id, IN_DISTRIBUTION_ITERATIONS[0]),
        (phase0.get_ood, OUT_OF_DISTRIBUTION_ITERATIONS[0]),
    ):
        fake.calls.clear()
        trace = getter(key).numpy()
        cfg = FewShotConfig(
            device="cpu",
            model_slug=FINETUNED_SLUG,
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=2,
            random_seed=0,
            example_target_length=None,
            normalization="shared",
            point_stat="mean",
            checkpoint="fake@000000000000",
        )
        out = fn(trace, pool[:2], cfg)
        expected = raw_param_tensor(get_params_for_benchmark_trace(key))
        assert out.shape == trace.shape and np.all(np.isfinite(out))
        assert fake.calls and all(torch.equal(c, expected) for c in fake.calls), (
            f"R4: conditioning tensor wrong for {key}"
        )
        assert ConditionRegistry.get(OP_PARAM_KEY) is None, "R4: registry not cleared"
    print("✓ R4: phase0 finetuned forecast fn resolves [0::3] query params (ID + OOD)")

    # R5 — end-to-end through run_benchmark on the Phase-0 provider
    cfg = FewShotConfig(
        device="cpu",
        model_slug=FINETUNED_SLUG,
        model_prediction_length=64,
        start_context_length=80,
        relevant_prediction_tail=80,
        k_shot=1,
        random_seed=0,
        example_target_length=None,
        normalization="shared",
        point_stat="mean",
        checkpoint="fake@000000000000",
    )
    results = run_benchmark(
        forecast_fn=make_phase0_finetuned_forecast_fn(fake, phase0, "mean", "shared"),
        config=cfg,
        example_pool=pool,
        method=f"{FINETUNED_SLUG.replace('/', '_')}_random__shared-mean-phase0",
        seeds=(0,),
        provider=phase0,
        save=False,
    )
    assert results.n_seeds == 1 and len(results.per_seed[0].per_trace) == 11
    assert np.isfinite(results.in_distribution.rmse)
    # Per-trace records must carry the [0::3] tail means (267-step traces)
    for tr in results.per_seed[0].per_trace:
        getter = phase0.get_ood if tr.trace_key.startswith("ood_") else phase0.get_id
        assert tr.true_tail_mean == float(np.mean(getter(tr.trace_key).numpy()[-80:]))
    print(
        f"✓ R5: run_benchmark on Phase0 provider (ID {results.in_distribution.rmse:.2f}, "
        f"OOD {results.out_of_distribution.rmse:.2f}); per-trace tails are [0::3]"
    )

    print("\n✅ Reconciliation self-tests passed!")
