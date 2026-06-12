"""Phase-7 mechanism dump: re-run headline cells, persist FULL forecasts.

Every few-shot results JSON (v2_selection..v6_finetuned) stores per-trace
SCALARS + ``example_ids`` only — no full forecasts anywhere. The Phase-7
mechanism analyses (error decomposition, rollout-stability/horizon, forecast
plots) need full trajectories, so this runner re-executes 13 headline cells
through the FROZEN harness with the ``forecast_callback`` hook and dumps
forecasts + ground truths per cell into ``results/few_shot_v7_mechanism/``.

Cells (all shared scaling + mean decoding unless noted; load order base
chronos2 -> bolt -> ft-full -> ft-512 -> CPU baselines — ONE finetuned model
per process at a time, never reusing an earlier ft pipeline):

==  =====================  =================  ===  ======  =========  =================
#   model                  strategy           k    window  seeds      consistency ref
==  =====================  =================  ===  ======  =========  =================
1   chronos2 base (bf16)   zeroshot           0    —       (42,)      v6 base mean
2   chronos2 base (bf16)   mmr_euclid         5    —       (42,)      v6 base mean
3   chronos2 base (bf16)   oracle_tail        10   —       (42,)      v6 base mean
4   chronos-bolt-tiny      zeroshot           0    —       (42,)      v5 bolt mean
5   chronos-bolt-tiny      mmr_euclid         10   —       (42,)      v5 bolt mean (22.63)
6   ft (fp32)              zeroshot           0    full    (42,)      v6 ft mean (22.20)
7   ft (fp32)              mmr_euclid         5    full    (42,)      v6 ft mean (18.62)
8   ft (fp32)              oracle_tail        10   full    (42,)      v6 ft mean (9.39)
9   ft (fp32)              random             10   full    (0,1,2)    v6 ft per_seed[0..2]
10  ft (fp32)              mmr_euclid         5    512     (42,)      v6 win512 (15.63)
11  ft (fp32)              oracle_tail        10   512     (42,)      v6 win512 (19.57)
12  baseline (CPU)         persistence        0    —       (42,)      internal asserts
13  baseline (CPU)         knn_copy_k5        0    —       (42,)      internal asserts
==  =====================  =================  ===  ======  =========  =================

JSON per cell: ``{ts}_{method}_k{k}_mechanism_dump.json`` — the suffix does
NOT match the ``*_fewshot_results.json`` glob, so the dumps stay invisible to
``harness.load_results`` (same trick as ``severin_anchor.json``). Content =
the full ``run_benchmark(save=False)`` result dict + per-seed ``forecasts``
+ top-level ``ground_truths`` + ``forecast_dtype`` + a ``consistency`` block
(reference path, n bit-equal, max rel diff).

HARD ASSERTS per cell with a v5/v6 reference (deterministic cells bit-
reproduce across process runs on this machine — verified 3x in Phase 6):
dumped ``mean(forecast[-80:])`` vs the recorded ``pred_tail_mean``
(rel < ``--rel-tol``, default 1e-6; bit-equality reported — if an env update
ever breaks bitwise reproduction, the documented fallback is
``--rel-tol 1e-3``), ``example_ids`` ≡ reference, context passthrough,
all-finite, ft checkpoint id ≡ the reference's recorded checkpoint.

Usage (smoke first):
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_mechanism_dump \
        --smoke --device mps
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_mechanism_dump \
        --device mps

``--smoke`` (loads the real base Chronos-2, writes nothing): S1 pool/provider
sanity, S2 dump-schema roundtrip through the real writer (constant CPU
forecast; filename must NOT match the results glob; stored tail means must
recompute bit-exactly under ``forecast_dtype``), S3 one real zero-shot trace
vs the v6 reference tail, S4 timing estimate for the full dump.
"""

import argparse
import fnmatch
import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from fusiontimeseries.benchmarking.few_shot.baselines import (
    make_knn_copy_forecast,
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
    make_finetuned_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.harness import (
    FewShotRunResults,
    ForecastFn,
    SelectFn,
    load_results,
    run_benchmark,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
from fusiontimeseries.benchmarking.few_shot.presentation import (
    make_concat_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    MODEL_SLUGS,
    PREDICT_FACTORIES,
    chronos2_predict_from_pipeline,
    make_chronos2_pipeline,
)
from fusiontimeseries.benchmarking.few_shot.run_presentation_grid import (
    variant_label,
)
from fusiontimeseries.benchmarking.few_shot.selection import STRATEGIES, make_select_fn
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    IN_DISTRIBUTION_ITERATIONS,
    OUT_OF_DISTRIBUTION_ITERATIONS,
    BenchmarkDataProvider,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v7_mechanism"
V5_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v5_decoding"
V6_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v6_finetuned"
DEFAULT_CHECKPOINT: Path = (
    REPO_ROOT / "outputs" / "chronos2-bilinear-selftrained-0" / "lora_weights.pt"
)

BASE_SLUG: str = MODEL_SLUGS["chronos2"]
BOLT_SLUG: str = MODEL_SLUGS["chronos_bolt"]
RANDOM_SEEDS: tuple[int, ...] = (0, 1, 2)


def method_label(slug: str, strategy: str, model_context_window: int | None = None) -> str:
    """v5/v6-compatible method label (shared scaling + mean decoding)."""
    variant = variant_label(
        normalization="shared",
        point_stat="mean",
        model_context_window=model_context_window,
    )
    return f"{slug.replace('/', '_')}_{strategy}__{variant}"


def dump_filename(timestamp: str, method: str, k: int) -> str:
    """Per-cell dump filename — must NOT match ``*_fewshot_results.json``."""
    name = f"{timestamp}_{method}_k{k}_mechanism_dump.json"
    assert not fnmatch.fnmatch(name, "*_fewshot_results.json")
    return name


def build_reference_index(results_dir: Path) -> dict[tuple[str, int], dict]:
    """(method, k_shot) -> latest result dict for a results directory."""
    index: dict[tuple[str, int], dict] = {}
    for r in load_results(results_dir):
        key = (r["method"], int(r["config"]["k_shot"]))
        if key not in index or r["timestamp"] > index[key]["timestamp"]:
            index[key] = r
    return index


def build_dump_payload(
    results: FewShotRunResults,
    captured: dict[tuple[int, str], NDArray],
    truths: dict[str, NDArray[np.float32]],
    consistency: dict,
) -> dict:
    """Results dict + per-seed forecasts + ground truths + consistency block."""
    dtypes = {str(forecast.dtype) for forecast in captured.values()}
    assert len(dtypes) == 1, f"Mixed forecast dtypes in one cell: {dtypes}"
    data = results.model_dump()
    for seed_data in data["per_seed"]:
        seed_data["forecasts"] = {
            tr["trace_key"]: captured[(seed_data["seed"], tr["trace_key"])].tolist()
            for tr in seed_data["per_trace"]
        }
    data["ground_truths"] = {key: trace.tolist() for key, trace in truths.items()}
    data["forecast_dtype"] = dtypes.pop()
    data["consistency"] = consistency
    return data


class MechanismDumpRunner:
    """Shared plumbing: pool, provider, selectors, reference indexes, writer."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.device: str = args.device
        self.save_dir: Path = args.save_dir
        self.rel_tol: float = args.rel_tol
        self.pool = create_example_pool(
            exclude_ids=set(ID_TEST_RAW_IDS), target_length=None
        )
        assert len(self.pool) == 245, f"Expected fixed pool of 245, got {len(self.pool)}"
        self.provider = BenchmarkDataProvider()
        self.truths: dict[str, NDArray[np.float32]] = {
            **{k: self.provider.get_id(k).numpy() for k in IN_DISTRIBUTION_ITERATIONS},
            **{
                k: self.provider.get_ood(k).numpy()
                for k in OUT_OF_DISTRIBUTION_ITERATIONS
            },
        }
        # Built ONCE so per-query ranking caches are shared across all cells.
        self.select_fns: dict[str, SelectFn] = {
            strategy: make_select_fn(strategy) for strategy in STRATEGIES
        }
        self.v5_index = build_reference_index(V5_RESULTS_DIR)
        self.v6_index = build_reference_index(V6_RESULTS_DIR)

    def reference(self, which: str, method: str, k: int) -> dict:
        index = {"v5": self.v5_index, "v6": self.v6_index}[which]
        ref = index.get((method, k))
        assert ref is not None, f"No {which} reference for {method} k={k}"
        return ref

    def cell_exists(self, method: str, k: int) -> bool:
        return any(self.save_dir.glob(f"*_{method}_k{k}_mechanism_dump.json"))

    def dump_cell(
        self,
        forecast_fn: ForecastFn,
        slug: str,
        strategy: str,
        k: int,
        method: str,
        select_fn: SelectFn | None,
        seeds: tuple[int, ...],
        deterministic: bool,
        reference: dict | None,
        checkpoint: str | None = None,
        model_context_window: int | None = None,
        normalization: str = "shared",
        point_stat: str = "mean",
        extra_check: Callable[[dict[tuple[int, str], NDArray], FewShotRunResults], None]
        | None = None,
    ) -> None:
        """Run one cell, hard-assert consistency, write the dump JSON."""
        if self.cell_exists(method, k):
            print(f"[skip] {method} k={k}: already in {self.save_dir.name}", flush=True)
            return
        if checkpoint is not None and reference is not None:
            ref_ck = reference["config"].get("checkpoint")
            assert ref_ck == checkpoint, (
                f"{method}: checkpoint {checkpoint} != reference's {ref_ck}"
            )
        config = FewShotConfig(
            device=self.device,
            model_slug=slug,
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=k,
            random_seed=seeds[0],
            example_target_length=None,
            normalization=normalization,
            point_stat=point_stat,
            checkpoint=checkpoint,
            model_context_window=model_context_window,
        )

        captured: dict[tuple[int, str], NDArray] = {}

        def capture(trace_key: str, seed: int, forecast: NDArray) -> None:
            captured[(seed, trace_key)] = np.asarray(forecast).copy()

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
            save=False,
            forecast_callback=capture,
        )
        assert len(captured) == len(results.seeds) * 11

        # Internal asserts: finite, context passthrough, tail single-sourcing
        for seed_result in results.per_seed:
            for tr in seed_result.per_trace:
                forecast = captured[(seed_result.seed, tr.trace_key)]
                truth = self.truths[tr.trace_key]
                assert forecast.shape == truth.shape, (
                    f"{method}/{tr.trace_key}: forecast shape {forecast.shape}"
                )
                assert np.all(np.isfinite(forecast)), (
                    f"{method}/{tr.trace_key}: non-finite forecast"
                )
                n_ctx = config.start_context_length
                assert np.array_equal(
                    forecast[:n_ctx].astype(np.float64),
                    truth[:n_ctx].astype(np.float64),
                ), f"{method}/{tr.trace_key}: context not passed through"
                tail = float(np.mean(forecast[-config.relevant_prediction_tail :]))
                assert tail == tr.pred_tail_mean, (
                    f"{method}/{tr.trace_key}: captured tail {tail!r} != "
                    f"recorded {tr.pred_tail_mean!r}"
                )
        if extra_check is not None:
            extra_check(captured, results)

        # Reference asserts: example_ids + recorded tails (bit-equality reported)
        consistency: dict = {
            "reference": None,
            "rel_tol": self.rel_tol,
            "n_compared": 0,
            "n_bit_equal": 0,
            "max_rel_diff": 0.0,
            "checks": [
                "finite",
                "context_passthrough",
                "tail_mean_single_sourced",
            ],
        }
        if reference is not None:
            consistency["reference"] = str(
                Path(reference["_path"]).relative_to(REPO_ROOT)
            )
            consistency["checks"] += ["example_ids", "tail_mean_vs_reference"]
            ref_per_seed = {sr["seed"]: sr for sr in reference["per_seed"]}
            for seed_result in results.per_seed:
                assert seed_result.seed in ref_per_seed, (
                    f"{method}: seed {seed_result.seed} missing in reference"
                )
                ref_seed = ref_per_seed[seed_result.seed]
                assert seed_result.example_ids == ref_seed["example_ids"], (
                    f"{method} seed {seed_result.seed}: example_ids != reference"
                )
                ref_tails = {
                    tr["trace_key"]: tr["pred_tail_mean"]
                    for tr in ref_seed["per_trace"]
                }
                for tr in seed_result.per_trace:
                    ref_tail = ref_tails[tr.trace_key]
                    rel = abs(tr.pred_tail_mean - ref_tail) / max(1.0, abs(ref_tail))
                    consistency["n_compared"] += 1
                    consistency["n_bit_equal"] += tr.pred_tail_mean == ref_tail
                    consistency["max_rel_diff"] = max(consistency["max_rel_diff"], rel)
                    assert rel < self.rel_tol, (
                        f"{method}/{tr.trace_key} seed {seed_result.seed}: tail "
                        f"{tr.pred_tail_mean:.6f} vs reference {ref_tail:.6f} "
                        f"(rel {rel:.2e} >= {self.rel_tol:.0e}) — if an env "
                        f"update broke bit-reproduction, re-run with --rel-tol 1e-3"
                    )

        payload = build_dump_payload(results, captured, self.truths, consistency)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.save_dir / dump_filename(results.timestamp, method, k)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        bit_note = (
            f"{consistency['n_bit_equal']}/{consistency['n_compared']} bit-equal, "
            f"max rel {consistency['max_rel_diff']:.1e}"
            if reference is not None
            else "internal asserts only"
        )
        print(
            f"[{method}] k={k}: ID {results.in_distribution.rmse:.2f}, "
            f"OOD {results.out_of_distribution.rmse:.2f} ({bit_note}) "
            f"[{time.perf_counter() - t0:.0f}s] -> {out_path.name}",
            flush=True,
        )

    def cleanup(self, *objects) -> None:
        del objects
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


########################################################
# Stages (fixed load order)
########################################################


def run_base_cells(runner: MechanismDumpRunner, args: argparse.Namespace) -> None:
    print(f"\n=== BASE {BASE_SLUG} (bf16) on {args.device} ===", flush=True)
    pipeline = make_chronos2_pipeline(args.device)
    forecast_fn = make_concat_forecast_fn(
        chronos2_predict_from_pipeline(pipeline, "mean"), "shared"
    )
    for strategy, k in (("zeroshot", 0), ("mmr_euclid", 5), ("oracle_tail", 10)):
        method = method_label(BASE_SLUG, strategy)
        runner.dump_cell(
            forecast_fn, BASE_SLUG, strategy, k, method,
            runner.select_fns.get(strategy), (42,), True,
            reference=runner.reference("v6", method, k),
        )
    runner.cleanup(pipeline, forecast_fn)


def run_bolt_cells(runner: MechanismDumpRunner, args: argparse.Namespace) -> None:
    print(f"\n=== BOLT {BOLT_SLUG} (bf16) on {args.device} ===", flush=True)
    predict_fn = PREDICT_FACTORIES["chronos_bolt"](args.device, "mean")
    forecast_fn = make_concat_forecast_fn(predict_fn, "shared")
    for strategy, k in (("zeroshot", 0), ("mmr_euclid", 10)):
        method = method_label(BOLT_SLUG, strategy)
        runner.dump_cell(
            forecast_fn, BOLT_SLUG, strategy, k, method,
            runner.select_fns.get(strategy), (42,), True,
            reference=runner.reference("v5", method, k),
        )
    runner.cleanup(predict_fn, forecast_fn)


def run_ft_cells(runner: MechanismDumpRunner, args: argparse.Namespace) -> None:
    ck = checkpoint_id(args.checkpoint)
    print(
        f"\n=== FINETUNED {FINETUNED_SLUG} (fp32, full window) on {args.device} "
        f"[{ck}] ===",
        flush=True,
    )
    pipeline = load_finetuned_chronos2(args.checkpoint, args.device)
    forecast_fn = make_finetuned_forecast_fn(pipeline, "mean", "shared")
    for strategy, k in (("zeroshot", 0), ("mmr_euclid", 5), ("oracle_tail", 10)):
        method = method_label(FINETUNED_SLUG, strategy)
        runner.dump_cell(
            forecast_fn, FINETUNED_SLUG, strategy, k, method,
            runner.select_fns.get(strategy), (42,), True,
            reference=runner.reference("v6", method, k), checkpoint=ck,
        )
    method = method_label(FINETUNED_SLUG, "random")
    runner.dump_cell(
        forecast_fn, FINETUNED_SLUG, "random", 10, method,
        runner.select_fns["random"], RANDOM_SEEDS, False,
        reference=runner.reference("v6", method, 10), checkpoint=ck,
    )
    runner.cleanup(pipeline, forecast_fn)


def run_ft512_cells(runner: MechanismDumpRunner, args: argparse.Namespace) -> None:
    ck = checkpoint_id(args.checkpoint)
    print(
        f"\n=== FINETUNED {FINETUNED_SLUG} (fp32, window {FT_TRAIN_CONTEXT}) on "
        f"{args.device} [{ck}] ===",
        flush=True,
    )
    pipeline = load_finetuned_chronos2(
        args.checkpoint, args.device, context_window=FT_TRAIN_CONTEXT
    )
    forecast_fn = make_finetuned_forecast_fn(pipeline, "mean", "shared")
    for strategy, k in (("mmr_euclid", 5), ("oracle_tail", 10)):
        method = method_label(FINETUNED_SLUG, strategy, FT_TRAIN_CONTEXT)
        runner.dump_cell(
            forecast_fn, FINETUNED_SLUG, strategy, k, method,
            runner.select_fns.get(strategy), (42,), True,
            reference=runner.reference("v6", method, k), checkpoint=ck,
            model_context_window=FT_TRAIN_CONTEXT,
        )
    runner.cleanup(pipeline, forecast_fn)


def run_baseline_cells(runner: MechanismDumpRunner) -> None:
    print("\n=== BASELINES (CPU, model-free) ===", flush=True)

    def constant_tail_check(level_fn) -> Callable:
        """Assert the post-context forecast is the expected constant level."""

        def check(
            captured: dict[tuple[int, str], NDArray], results: FewShotRunResults
        ) -> None:
            for (seed, trace_key), forecast in captured.items():
                expected = level_fn(runner.truths[trace_key])
                assert np.all(forecast[80:] == expected), (
                    f"baseline/{trace_key}: tail not the expected constant level"
                )

        return check

    runner.dump_cell(
        persistence_forecast, "baseline", "persistence", 0, "persistence",
        None, (42,), True, reference=None,
        normalization="per_example", point_stat="median",
        extra_check=constant_tail_check(lambda trace: np.float32(trace[79])),
    )
    knn_fn = make_knn_copy_forecast(runner.pool, k=5, rescale=False)

    def knn_constant_check(
        captured: dict[tuple[int, str], NDArray], results: FewShotRunResults
    ) -> None:
        for (seed, trace_key), forecast in captured.items():
            tail = forecast[80:]
            assert np.all(tail == tail[0]), f"knn_copy/{trace_key}: tail not constant"

    runner.dump_cell(
        knn_fn, "baseline", "knn_copy_k5", 0, "knn_copy_k5",
        None, (42,), True, reference=None,
        normalization="per_example", point_stat="median",
        extra_check=knn_constant_check,
    )


########################################################
# Smoke (loads the real base model, writes nothing)
########################################################


def smoke(runner: MechanismDumpRunner, args: argparse.Namespace) -> None:
    import tempfile

    print(f"\n=== Smoke: mechanism-dump checks on {args.device} ===", flush=True)

    # S1 — pool/provider sanity
    assert len(runner.truths) == 11
    assert all(trace.shape == (266,) for trace in runner.truths.values())
    assert len(runner.v6_index) >= 30 and len(runner.v5_index) >= 30
    print("✓ S1: pool 245, 11 benchmark traces of length 266, v5/v6 indexes loaded")

    # S2 — schema roundtrip through the real writer (constant CPU forecast)
    with tempfile.TemporaryDirectory() as tmp:
        config = FewShotConfig(
            device="cpu", model_slug="baseline", model_prediction_length=64,
            start_context_length=80, relevant_prediction_tail=80, k_shot=0,
            random_seed=42, example_target_length=None,
        )
        captured: dict[tuple[int, str], NDArray] = {}
        results = run_benchmark(
            forecast_fn=persistence_forecast, config=config,
            example_pool=runner.pool, method="smoke_persistence", seeds=(42,),
            deterministic=True, provider=runner.provider, save=False,
            forecast_callback=lambda key, seed, fc: captured.__setitem__(
                (seed, key), np.asarray(fc).copy()
            ),
        )
        payload = build_dump_payload(
            results, captured, runner.truths,
            {"reference": None, "n_compared": 0, "n_bit_equal": 0,
             "max_rel_diff": 0.0, "rel_tol": args.rel_tol, "checks": []},
        )
        name = dump_filename(results.timestamp, "smoke_persistence", 0)
        out_path = Path(tmp) / name
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        loaded = json.load(open(out_path))
        assert not fnmatch.fnmatch(name, "*_fewshot_results.json")
        assert set(loaded["ground_truths"]) == set(runner.truths)
        dtype = np.dtype(loaded["forecast_dtype"])
        for seed_data in loaded["per_seed"]:
            for tr in seed_data["per_trace"]:
                stored = np.array(
                    seed_data["forecasts"][tr["trace_key"]], dtype=dtype
                )
                recomputed = float(np.mean(stored[-80:]))
                assert recomputed == tr["pred_tail_mean"], (
                    f"S2: stored forecast tail does not recompute bit-exactly "
                    f"for {tr['trace_key']}: {recomputed!r} vs "
                    f"{tr['pred_tail_mean']!r}"
                )
    print(
        "✓ S2: dump schema roundtrip (writer -> JSON -> reload); filename "
        "invisible to load_results; tail means recompute bit-exactly under "
        f"forecast_dtype={loaded['forecast_dtype']}"
    )

    # S3 — one real zero-shot trace vs the v6 reference
    ref = runner.reference("v6", method_label(BASE_SLUG, "zeroshot"), 0)
    ref_tail = {
        tr["trace_key"]: tr["pred_tail_mean"]
        for tr in ref["per_seed"][0]["per_trace"]
    }["iteration_8_ifft"]
    pipeline = make_chronos2_pipeline(args.device)
    forecast_fn = make_concat_forecast_fn(
        chronos2_predict_from_pipeline(pipeline, "mean"), "shared"
    )
    config0 = FewShotConfig(
        device=args.device, model_slug=BASE_SLUG, model_prediction_length=64,
        start_context_length=80, relevant_prediction_tail=80, k_shot=0,
        random_seed=42, example_target_length=None, normalization="shared",
        point_stat="mean",
    )
    trace = runner.truths["iteration_8_ifft"]
    t0 = time.perf_counter()
    forecast = forecast_fn(trace, [], config0)
    dt = time.perf_counter() - t0
    tail = float(np.mean(forecast[-80:]))
    rel = abs(tail - ref_tail) / max(1.0, abs(ref_tail))
    assert rel < 1e-3, f"S3: zero-shot drifted vs v6: {tail:.4f} vs {ref_tail:.4f}"
    print(
        f"✓ S3: base zero-shot iteration_8_ifft tail {tail:.4f} vs v6 "
        f"{ref_tail:.4f} ({'bit-equal' if tail == ref_tail else f'rel {rel:.1e}'})"
    )

    # S4 — timing estimate: 12 deterministic cells x 11 + 1 random cell x 33
    n_passes = 12 * 11 + len(RANDOM_SEEDS) * 11
    print(
        f"✓ S4: single trace pass {dt:.1f}s -> full dump ≈ "
        f"{n_passes * dt / 60:.0f} min ({n_passes} trace passes; ft cells are "
        f"fp32 and somewhat slower)"
    )
    runner.cleanup(pipeline, forecast_fn)
    print("\n✅ Smoke test passed (no dumps written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-7 mechanism forecast dump")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="lora_weights.pt for the finetuned cells",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--rel-tol", type=float, default=1e-6,
        help="Hard-assert tolerance vs recorded v5/v6 tails (fallback 1e-3 "
        "if an env update ever breaks bit-reproduction)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run the S1-S4 sanity checks and exit (writes nothing)",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()

    runner = MechanismDumpRunner(args)
    if args.smoke:
        smoke(runner, args)
        return

    assert args.checkpoint.exists(), f"No checkpoint at {args.checkpoint}"
    t0 = time.perf_counter()
    run_base_cells(runner, args)
    run_bolt_cells(runner, args)
    run_ft_cells(runner, args)
    run_ft512_cells(runner, args)
    run_baseline_cells(runner)
    print(
        f"\n✅ Mechanism dump complete -> {args.save_dir} "
        f"[{(time.perf_counter() - t0) / 60:.0f} min]",
        flush=True,
    )


if __name__ == "__main__":
    main()
