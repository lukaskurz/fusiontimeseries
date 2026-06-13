"""Phase-6 grid: ICL on top of the finetuned Chronos-2 BilinearLoRA.

Does adaptation STACK? The deliverable is the 2x2 {base, finetuned} x
{k=0, k-best} table plus paired comparisons. All cells use Phase-3's winning
presentation (flat concat + SHARED scaling), t266 protocol, fixed 245-trace
pool, both decodings ({median, mean} — mean is the v5 Chronos-2 default,
median is the bridge twin), ONE save dir
(``results/few_shot_v6_finetuned/``) so headline pairs live within a single
grid run. Fresh base twins are run in-dir; the base configs overlapping v5
(zeroshot / mmr_euclid k=5 / oracle_tail k=10 / random k=10) double as the
cross-dir bridge. Base = bf16 (protocol continuity with v3-v5), finetuned =
fp32 (training numerics) — noted in the analysis captions.

Stages (40 cells + the anchor JSON):

- ``anchors``: {base, ft} x zeroshot k=0 x {median, mean}            = 4
- ``icl``:     {base, ft} x {ctx_euclid k5/k10, mmr_euclid k5/k10,
               oracle_tail k10, op_knn k5/k10} x {median, mean}      = 28
- ``random``:  {base, ft} x random k=10 (20 seeds) x {median, mean}  = 4
- ``window``:  ft @ chronos_config 512 (the TRAINING window;
               ``pipeline.predict`` clamps the ICL stream down to it)
               x {mmr_euclid k5, k10} x {median, mean}               = 4
- ``anchor``:  ``severin_anchor_eval`` (his ``mean(x[:-80])`` metric AND the
               honest ``mean(x[-80:])`` on the same forecasts) ->
               ``severin_anchor.json`` (name does not match the results
               glob, so it stays invisible to ``load_results``).

Finetuned cells carry ``checkpoint`` (``name@sha256[:12]``) in their config
— Severin's ``lora_weights.pt`` can be swapped in via ``--checkpoint`` for a
minutes-long re-run. ``BilinearLoRA._shared_p_projection`` is a CLASS
attribute: the runner loads ONE finetuned model at a time and never uses an
earlier load after a later one.

Usage (smoke first — F2b is the go/no-go):
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_finetuned_grid \
        --checkpoint outputs/chronos2-bilinear-selftrained-0/lora_weights.pt \
        --smoke --device mps
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_finetuned_grid \
        --checkpoint outputs/chronos2-bilinear-selftrained-0/lora_weights.pt \
        --device mps

``--smoke`` (real models, writes nothing): F1 checkpoint load sanity (exact
LoRA key-set + shape asserts inside the loader; no-patch forward raises),
F2 conditioning liveness (two param sets diverge), F2b param-order gate
(all 11 benchmark traces: resolver params reordered [shat, q, rlt, rln] must
EXACTLY equal the FluxData entries' operating_parameters — go/no-go),
F3 pipeline-vs-raw-forward k=0 consistency (rel L2, loose tol),
F4 base zero-shot regression vs the v5 JSON (bit-equality reported),
F5 window knob (8192/512; ft k=0 invariant, k=5 stream differs),
F6 timing (one ft mean k=10 pass -> per-cell estimate).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import FewShotConfig
from fusiontimeseries.benchmarking.few_shot.finetuned import (
    BASE_CONTEXT_WINDOW,
    FINETUNED_SLUG,
    FT_TRAIN_CONTEXT,
    checkpoint_id,
    load_finetuned_chronos2,
    make_finetuned_forecast_fn,
    raw_param_tensor,
    severin_anchor_eval,
)
from fusiontimeseries.benchmarking.few_shot.harness import ForecastFn, load_results
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    get_params_for_benchmark_trace,
)
from fusiontimeseries.benchmarking.few_shot.presentation import (
    make_concat_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    MODEL_SLUGS,
    chronos2_predict_from_pipeline,
    make_chronos2_pipeline,
)
from fusiontimeseries.benchmarking.few_shot.run_presentation_grid import (
    CellRunner,
    seeds_for,
)
from fusiontimeseries.lib.conditioning import ConditionRegistry

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v6_finetuned"
V5_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v5_decoding"

STAGES: tuple[str, ...] = ("anchors", "icl", "random", "window", "anchor")
POINT_STATS: tuple[str, ...] = ("median", "mean")
ICL_CONFIGS: tuple[tuple[str, int], ...] = (
    ("ctx_euclid", 5),
    ("ctx_euclid", 10),
    ("mmr_euclid", 5),
    ("mmr_euclid", 10),
    ("oracle_tail", 10),
    # Phase-7 oracle-gap probe (2026-06-13): op_knn re-tested ON THE FINETUNED
    # model — uniquely motivated there (the ft model is literally conditioned
    # on those params; Phase-2's "op_knn ≈ ctx" verdict was base-model only).
    ("op_knn", 5),
    ("op_knn", 10),
    # Part A (2026-06-13): level-aware retrieval on the finetuned model.
    # ctx_level (match absolute context level — the signal the tail-mean metric
    # scores) and mmr_level (level relevance + shape diversity) postdate Phase 6
    # (which only had mmr_euclid shape-matching). The model-free + Bolt
    # level_matching follow-up showed level-matching beats shape-matching
    # dramatically OOD; this tests whether that stacks on the ft model.
    ("ctx_level", 5),
    ("ctx_level", 10),
    ("mmr_level", 5),
    ("mmr_level", 10),
)
WINDOW_CONFIGS: tuple[tuple[str, int], ...] = (
    ("mmr_euclid", 5),
    ("mmr_euclid", 10),
    # Robustness cell B (2026-06-13): the cheating ceiling at the training
    # window — does clamping help even when the examples are oracle-picked?
    ("oracle_tail", 10),
    # Part A (2026-06-13): level-aware retrieval @ the 512 training window.
    ("ctx_level", 5),
    ("ctx_level", 10),
    ("mmr_level", 5),
    ("mmr_level", 10),
)
RANDOM_K: int = 10


def run_model_cells(
    runner: CellRunner,
    args: argparse.Namespace,
    slug: str,
    make_forecast_fn,
    checkpoint: str | None,
) -> None:
    """anchors/icl/random cells for one loaded model (base or ft)."""
    for point_stat in POINT_STATS:
        forecast_fn: ForecastFn = make_forecast_fn(point_stat)
        if "anchors" in args.stages:
            runner.run_cell(
                forecast_fn, slug, "zeroshot", 0, None, (42,), True,
                normalization="shared", point_stat=point_stat, checkpoint=checkpoint,
            )
        if "icl" in args.stages:
            for strategy, k in ICL_CONFIGS:
                seeds, deterministic = seeds_for(strategy, args.random_seeds)
                runner.run_cell(
                    forecast_fn, slug, strategy, k, runner.select_fns[strategy],
                    seeds, deterministic,
                    normalization="shared", point_stat=point_stat, checkpoint=checkpoint,
                )
        if "random" in args.stages:
            runner.run_cell(
                forecast_fn, slug, "random", RANDOM_K, runner.select_fns["random"],
                tuple(range(args.random_seeds)), False,
                normalization="shared", point_stat=point_stat, checkpoint=checkpoint,
            )
        runner.cleanup(forecast_fn)


def run_base(runner: CellRunner, args: argparse.Namespace) -> None:
    slug = MODEL_SLUGS["chronos2"]
    print(f"\n=== BASE {slug} (bf16) on {args.device} ===", flush=True)
    pipeline = make_chronos2_pipeline(args.device)
    run_model_cells(
        runner, args, slug,
        lambda ps: make_concat_forecast_fn(
            chronos2_predict_from_pipeline(pipeline, ps), "shared"
        ),
        checkpoint=None,
    )
    runner.cleanup(pipeline)


def run_finetuned(runner: CellRunner, args: argparse.Namespace) -> None:
    ck = checkpoint_id(args.checkpoint)
    print(
        f"\n=== FINETUNED {FINETUNED_SLUG} (fp32, window {BASE_CONTEXT_WINDOW}) "
        f"on {args.device} [{ck}] ===",
        flush=True,
    )
    pipeline = load_finetuned_chronos2(args.checkpoint, args.device)
    run_model_cells(
        runner, args, FINETUNED_SLUG,
        lambda ps: make_finetuned_forecast_fn(pipeline, ps, "shared"),
        checkpoint=ck,
    )
    runner.cleanup(pipeline)


def run_window(runner: CellRunner, args: argparse.Namespace) -> None:
    """ft @ the 512 TRAINING window — pipeline.predict clamps the stream."""
    ck = checkpoint_id(args.checkpoint)
    print(
        f"\n=== FINETUNED {FINETUNED_SLUG} (fp32, window {FT_TRAIN_CONTEXT}) "
        f"on {args.device} [{ck}] ===",
        flush=True,
    )
    pipeline = load_finetuned_chronos2(
        args.checkpoint, args.device, context_window=FT_TRAIN_CONTEXT
    )
    for point_stat in POINT_STATS:
        forecast_fn = make_finetuned_forecast_fn(pipeline, point_stat, "shared")
        for strategy, k in WINDOW_CONFIGS:
            seeds, deterministic = seeds_for(strategy, args.random_seeds)
            runner.run_cell(
                forecast_fn, FINETUNED_SLUG, strategy, k,
                runner.select_fns[strategy], seeds, deterministic,
                normalization="shared", point_stat=point_stat, checkpoint=ck,
                model_context_window=FT_TRAIN_CONTEXT,
            )
        runner.cleanup(forecast_fn)
    runner.cleanup(pipeline)


def run_anchor(args: argparse.Namespace) -> None:
    """Severin-protocol eval (both metrics) -> severin_anchor.json."""
    ck = checkpoint_id(args.checkpoint)
    print(f"\n=== ANCHOR severin_anchor_eval on {args.device} [{ck}] ===", flush=True)
    pipeline = load_finetuned_chronos2(
        args.checkpoint, args.device, context_window=FT_TRAIN_CONTEXT
    )
    result = severin_anchor_eval(
        pipeline.model, args.device, severin_results_path=args.severin_results
    )
    result["checkpoint"] = ck
    result["device"] = args.device
    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.save_dir / "severin_anchor.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    sev, tail = result["metrics_severin_headminus80"], result["metrics_tail80"]
    print(
        f"[anchor] his metric mean(x[:-80]): ID {sev['id']['rmse']:.2f} / "
        f"OOD {sev['ood']['rmse']:.2f} (README BilinearLoRA row: 13.83 / 4.86)\n"
        f"[anchor] honest mean(x[-80:]):     ID {tail['id']['rmse']:.2f} / "
        f"OOD {tail['ood']['rmse']:.2f}\n"
        f"Wrote {out_path}",
        flush=True,
    )
    del pipeline
    if args.device == "mps":
        torch.mps.empty_cache()


########################################################
# Smoke (real models, writes nothing)
########################################################


def shared_scaled_stream(
    trace: np.ndarray, examples, k: int
) -> np.ndarray:
    """k-shot shared-scaled concat stream (mirrors make_concat_forecast_fn)."""
    scaler = StandardScaler().fit(trace[:80].reshape(-1, 1))
    segments = []
    for ex in examples[:k]:
        segments.append(scaler.transform(ex.context_array.reshape(-1, 1)).squeeze())
        segments.append(scaler.transform(ex.target_array.reshape(-1, 1)).squeeze())
    segments.append(scaler.transform(trace[:80].reshape(-1, 1)).squeeze())
    return np.concatenate(segments).astype(np.float32)


def smoke(runner: CellRunner, args: argparse.Namespace) -> None:
    print(f"\n=== Smoke: finetuned-grid checks on {args.device} ===", flush=True)
    provider = runner.provider
    trace = provider.get_id("iteration_8_ifft").numpy()

    # F2b — param-order gate (pure data, go/no-go before anything else):
    # resolver params in FluxData order must EXACTLY equal the FluxData
    # entries' operating_parameters for all 11 benchmark traces.
    from fusiontimeseries.finetuning.chronos2.dataset import Chronos2Dataset
    from fusiontimeseries.finetuning.chronos2.train_bilinear import (
        build_fts_config,
        ensure_flat_flux_data,
    )
    from fusiontimeseries.benchmarking.few_shot.operating_params import load_mapping

    fts_config = build_fts_config("cpu", max_steps=4000)
    fts_config.data_path = ensure_flat_flux_data()
    benchmark_data = Chronos2Dataset.get_benchmark_flux_traces(fts_config)
    mapping = load_mapping()["benchmark_traces"]
    checked = 0
    for trace_key, entry in mapping.items():
        if entry["kind"] == "id":
            flux_idx = entry["raw_id"]
        else:
            flux_idx = int(entry["dump_key"]) - 4000
        flux_data = benchmark_data[entry["kind"]][flux_idx]
        ours = raw_param_tensor(get_params_for_benchmark_trace(trace_key)).numpy()[0]
        theirs = flux_data.operating_parameters
        assert np.array_equal(ours, theirs.astype(np.float32)), (
            f"F2b FAILED — STOP: param order/values for {trace_key} "
            f"(flux {flux_idx}): ours {ours} vs FluxData {theirs}"
        )
        checked += 1
    assert checked == 11
    print(f"✓ F2b (go/no-go): conditioning vectors ≡ FluxData.operating_parameters for all 11 benchmark traces")

    # F4 — base regression: zero-shot median vs the v5 JSON per-trace tails
    v5_zeroshot = [
        r for r in load_results(V5_RESULTS_DIR)
        if r["method"] == f"{MODEL_SLUGS['chronos2'].replace('/', '_')}_zeroshot__shared"
        and r["config"].get("k_shot") == 0
    ]
    assert v5_zeroshot, f"No v5 chronos2 zeroshot__shared cell in {V5_RESULTS_DIR}"
    v5_ref = max(v5_zeroshot, key=lambda r: r["timestamp"])
    v5_tails = {
        tr["trace_key"]: tr["pred_tail_mean"]
        for tr in v5_ref["per_seed"][0]["per_trace"]
    }
    base_pipeline = make_chronos2_pipeline(args.device)
    base_fn = make_concat_forecast_fn(
        chronos2_predict_from_pipeline(base_pipeline, "median"), "shared"
    )
    config0 = FewShotConfig(
        device=args.device, model_slug=MODEL_SLUGS["chronos2"],
        model_prediction_length=64, start_context_length=80,
        relevant_prediction_tail=80, k_shot=0, random_seed=42,
        example_target_length=None, normalization="shared", point_stat="median",
    )
    bit_equal, max_rel = True, 0.0
    for trace_key, ref_tail in v5_tails.items():
        getter = provider.get_ood if trace_key.startswith("ood_") else provider.get_id
        t = getter(trace_key).numpy()
        tail = float(np.mean(base_fn(t, [], config0)[-80:]))
        rel = abs(tail - ref_tail) / max(1.0, abs(ref_tail))
        bit_equal &= tail == ref_tail
        max_rel = max(max_rel, rel)
        assert rel < 1e-3, (
            f"F4: base zero-shot drifted vs v5 for {trace_key}: "
            f"{tail:.4f} vs {ref_tail:.4f} (rel {rel:.2e})"
        )
    print(
        f"✓ F4: base zero-shot ≡ v5 per-trace tails "
        f"({'bit-equal' if bit_equal else f'max rel diff {max_rel:.2e}'}, 11 traces)"
    )
    runner.cleanup(base_pipeline, base_fn)

    # F1 — checkpoint load sanity (key/shape asserts live in the loader)
    ck = checkpoint_id(args.checkpoint)
    pipeline = load_finetuned_chronos2(args.checkpoint, args.device)
    assert pipeline.model.chronos_config.context_length == BASE_CONTEXT_WINDOW
    try:
        pipeline.predict(inputs=torch.randn(1, 1, 80), prediction_length=16)
        raise AssertionError("F1: no-patch predict must raise RuntimeError")
    except RuntimeError:
        pass
    print(f"✓ F1: {ck} loads (exact LoRA key set + shapes); no-patch forward raises")

    # F2 — conditioning liveness: two param sets diverge on the same context
    ctx = shared_scaled_stream(trace, [], 0)
    predict = chronos2_predict_from_pipeline(pipeline, "median")
    params_a = {"q": 2.0, "shat": 0.5, "rlt": 6.0, "rln": 2.0}
    params_b = {"q": 8.0, "shat": 4.0, "rlt": 11.0, "rln": 6.0}
    with ConditionRegistry.patch(op_params=raw_param_tensor(params_a)):
        out_a = np.asarray(predict(ctx, 64))
    with ConditionRegistry.patch(op_params=raw_param_tensor(params_b)):
        out_b = np.asarray(predict(ctx, 64))
    assert not np.array_equal(out_a, out_b), "F2: conditioning inert"
    print(
        f"✓ F2: conditioning live (param sets diverge, mean |Δ| "
        f"{float(np.mean(np.abs(out_a - out_b))):.4f} normalized)"
    )

    # F3 — pipeline vs raw forward, k=0, one 16-step block (loose tol: the
    # pipeline's preprocessing differs from the notebook's NaN-pad-512 path)
    params = get_params_for_benchmark_trace("iteration_8_ifft")
    op = raw_param_tensor(params)
    with ConditionRegistry.patch(op_params=op):
        pipe_out = np.asarray(predict(ctx, 16))
    raw_ctx = torch.full((1, FT_TRAIN_CONTEXT), float("nan"))
    raw_ctx[0, -len(ctx):] = torch.tensor(ctx)
    mask = torch.zeros_like(raw_ctx)
    mask[0, -len(ctx):] = 1.0
    with torch.no_grad(), ConditionRegistry.patch(op_params=op.to(args.device)):
        output = pipeline.model(
            context=raw_ctx.to(args.device), context_mask=mask.to(args.device)
        )
    q = output.quantile_preds
    raw_out = q[:, q.shape[1] // 2, :].cpu().numpy().flatten()
    rel = float(np.linalg.norm(pipe_out - raw_out) / max(1e-9, np.linalg.norm(raw_out)))
    assert rel < 0.25, f"F3: pipeline vs raw forward rel L2 {rel:.3f} >= 0.25"
    print(f"✓ F3: pipeline ≈ raw notebook forward (k=0, 16 steps, rel L2 {rel:.4f})")

    # F5 — window knob: k=0 invariant (stream 80 < 512), k=5 stream differs
    examples5 = runner.select_fns["mmr_euclid"](
        runner.pool, 5, 42, trace[:80], "iteration_8_ifft"
    )
    stream0 = shared_scaled_stream(trace, [], 0)
    stream5 = shared_scaled_stream(trace, examples5, 5)
    assert len(stream5) > FT_TRAIN_CONTEXT
    with ConditionRegistry.patch(op_params=op):
        full_k0 = np.asarray(predict(stream0, 64))
        full_k5 = np.asarray(predict(stream5, 64))
    runner.cleanup(pipeline, predict)
    pipeline512 = load_finetuned_chronos2(
        args.checkpoint, args.device, context_window=FT_TRAIN_CONTEXT
    )
    assert pipeline512.model.chronos_config.context_length == FT_TRAIN_CONTEXT
    predict512 = chronos2_predict_from_pipeline(pipeline512, "median")
    with ConditionRegistry.patch(op_params=op):
        win_k0 = np.asarray(predict512(stream0, 64))
        win_k5 = np.asarray(predict512(stream5, 64))
    assert np.allclose(full_k0, win_k0, rtol=0, atol=1e-6), (
        f"F5: ft k=0 not window-invariant (max |Δ| {np.max(np.abs(full_k0 - win_k0)):.2e})"
    )
    assert not np.array_equal(full_k5, win_k5), (
        "F5: k=5 stream identical under 8192 vs 512 — clamp inert?"
    )
    k0_note = "bit-equal" if np.array_equal(full_k0, win_k0) else "allclose"
    print(
        f"✓ F5: windows 8192/512 set; ft k=0 invariant ({k0_note}), "
        f"k=5 stream clamps (mean |Δ| {float(np.mean(np.abs(full_k5 - win_k5))):.4f})"
    )

    # F6 — timing: one ft mean k=10 full-trace rollout
    forecast_fn = make_finetuned_forecast_fn(pipeline512, "mean", "shared")
    examples10 = runner.select_fns["ctx_euclid"](
        runner.pool, 10, 42, trace[:80], "iteration_8_ifft"
    )
    config10 = FewShotConfig(
        device=args.device, model_slug=FINETUNED_SLUG,
        model_prediction_length=64, start_context_length=80,
        relevant_prediction_tail=80, k_shot=10, random_seed=42,
        example_target_length=None, normalization="shared", point_stat="mean",
        checkpoint=ck,
    )
    t0 = time.perf_counter()
    out = forecast_fn(trace, examples10, config10)
    dt = time.perf_counter() - t0
    assert np.all(np.isfinite(out)), "F6: non-finite ft k=10 forecast"
    print(
        f"✓ F6: ft mean k=10 single trace {dt:.1f}s -> det cell ≈ "
        f"{11 * dt / 60:.1f} min, {args.random_seeds}-seed random cell ≈ "
        f"{11 * args.random_seeds * dt / 60:.0f} min"
    )
    runner.cleanup(pipeline512, predict512, forecast_fn)
    print("\n✅ Smoke test passed (no results written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-6 finetuned-ICL grid")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="lora_weights.pt to reconstruct the finetuned model from",
    )
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=list(STAGES),
        help="Which stages to run",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--random-seeds", type=int, default=20,
        help="Number of seeds (0..n-1) for the random stage",
    )
    parser.add_argument(
        "--severin-results", type=Path, default=None,
        help="Optional benchmark_results.json from Severin's run (anchor stage "
        "reports per-trace drift against it)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run the F1-F6 sanity checks and exit (writes nothing)",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    assert args.checkpoint.exists(), f"No checkpoint at {args.checkpoint}"

    runner = CellRunner(args)
    if args.smoke:
        smoke(runner, args)
        return

    t0 = time.perf_counter()
    if {"anchors", "icl", "random"} & set(args.stages):
        run_base(runner, args)
        run_finetuned(runner, args)
    if "window" in args.stages:
        run_window(runner, args)
    if "anchor" in args.stages:
        run_anchor(args)
    print(
        f"\n✅ Finetuned grid stage(s) complete "
        f"[{(time.perf_counter() - t0) / 60:.0f} min]",
        flush=True,
    )


if __name__ == "__main__":
    main()
