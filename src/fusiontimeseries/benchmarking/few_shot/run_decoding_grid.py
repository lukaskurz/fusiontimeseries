"""Phase-5 decoding grid: point-statistic (median vs mean) x configs x models.

The benchmark metric is tail RMSE of a positive, right-skewed flux — the
RMSE-optimal point forecast is the conditional MEAN, but every Phase-1..4
cell decoded the MEDIAN (q0.5 of the 9 deciles). This grid re-runs anchors,
the best Phase-3 configs, an oracle ceiling, and 20-seed random cells under
both decodings; seed- and cross-model ensembling are computed post-hoc by
``analyze_decoding.py`` from the per-trace tail means already in the JSONs
(the harness stays frozen).

Decodings per model: ``median`` (the frozen path) and ``mean`` (decile
average over the 9 quantiles — truncates tails beyond q0.1/q0.9, hence a
biased-low "decile-average mean" on right-skewed data); TimesFM additionally
runs ``meanhead`` (its native mean output head, index 0 of the
``[mean, q0.1..q0.9]`` full forecast — the unbiased-head cross-check).
NOTE: TiRex has NO native mean — the library's second ``forecast()`` return
is a relabeled median (``# median as mean`` in ``tirex/models/tirex.py``).

Every cell uses Phase-3's winning presentation (flat concat + SHARED
scaling), t266 protocol, fixed 245-trace pool. All cells write to ONE save
dir (``results/few_shot_v5_decoding/``) so every headline comparison lives
within a single grid run (MPS is not bit-deterministic across runs). Stages
(each = 4 models x {median, mean} + TimesFM meanhead = 9 cells):

- ``anchors``: zero-shot k=0 — isolates the pure decoding effect including
  its feedback through the autoregressive rollout (the decoded point is fed
  back, so mean decoding changes the whole trajectory, not just the read-out).
- ``best``:   the best legit Phase-3 config per model — Bolt mmr_euclid
  k=10, TimesFM mmr_euclid k=10, TiRex ctx_euclid k=10, Chronos-2
  mmr_euclid k=5.
- ``oracle``: oracle_tail k=10 (cheating ceiling: does mean move it?).
- ``random``: random k=10, 20 seeds — the seed-ensembling cells.

= 36 cells. Method labels: ``{slug}_{strategy}__shared`` |
``__shared-mean`` | ``__shared-meanhead``. Same select_fn + same seeds per
decoding twin => identical example sets (the analyzer hard-asserts this).
A cell whose result file already exists in the save dir is skipped.

Usage (single background run; smoke first):
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_decoding_grid \
        --smoke --device mps
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_decoding_grid \
        --device mps

``--smoke`` loads the real models and runs, without writing results:
D1 median-path identity / in-process bit-reproducibility per model (+ the
TiRex library-"mean" ≡ median check), D2 TimesFM output-layout checks (the
go/no-go gate: first return ≡ index 5 of the full forecast; deciles
monotone; meanhead distinct), D3 estimator sanity (pure numpy/torch, no
model), D4 one mean-decoded k=10 timing pass per model.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import FewShotConfig
from fusiontimeseries.benchmarking.few_shot.presentation import make_concat_forecast_fn
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    MODEL_SLUGS,
    PREDICT_FACTORIES,
    decode_point_forecast,
)
from fusiontimeseries.benchmarking.few_shot.run_presentation_grid import (
    CellRunner,
    seeds_for,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import Utils

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v5_decoding"

STAGES: tuple[str, ...] = ("anchors", "best", "oracle", "random")

#: Best legit Phase-3 config per model (shared scaling, t266): model -> (strategy, k).
BEST_CONFIGS: dict[str, tuple[str, int]] = {
    "chronos_bolt": ("mmr_euclid", 10),
    "timesfm": ("mmr_euclid", 10),
    "tirex": ("ctx_euclid", 10),
    "chronos2": ("mmr_euclid", 5),
}
ORACLE_K: int = 10
RANDOM_K: int = 10


def point_stats_for(model_name: str) -> tuple[str, ...]:
    """Decodings per model; only TimesFM has a native mean head."""
    if model_name == "timesfm":
        return ("median", "mean", "meanhead")
    return ("median", "mean")


def run_model_stages(
    runner: CellRunner, model_name: str, args: argparse.Namespace
) -> None:
    """All selected stages for one model, one decoding at a time in memory."""
    slug = MODEL_SLUGS[model_name]
    for point_stat in point_stats_for(model_name):
        print(
            f"\n=== {slug} [{point_stat}] on {args.device}, stages={args.stages} ===",
            flush=True,
        )
        predict_fn = PREDICT_FACTORIES[model_name](args.device, point_stat)
        forecast_fn = make_concat_forecast_fn(predict_fn, "shared")

        if "anchors" in args.stages:
            runner.run_cell(
                forecast_fn, slug, "zeroshot", 0, None, (42,), True,
                normalization="shared", point_stat=point_stat,
            )
        if "best" in args.stages:
            strategy, k = BEST_CONFIGS[model_name]
            seeds, deterministic = seeds_for(strategy, args.random_seeds)
            runner.run_cell(
                forecast_fn, slug, strategy, k, runner.select_fns[strategy],
                seeds, deterministic,
                normalization="shared", point_stat=point_stat,
            )
        if "oracle" in args.stages:
            runner.run_cell(
                forecast_fn, slug, "oracle_tail", ORACLE_K,
                runner.select_fns["oracle_tail"], (42,), True,
                normalization="shared", point_stat=point_stat,
            )
        if "random" in args.stages:
            runner.run_cell(
                forecast_fn, slug, "random", RANDOM_K,
                runner.select_fns["random"], tuple(range(args.random_seeds)), False,
                normalization="shared", point_stat=point_stat,
            )
        runner.cleanup(predict_fn, forecast_fn)


########################################################
# Smoke test (real models, no results written)
########################################################


def smoke_estimator_sanity() -> None:
    """D3 — pure numpy/torch: helper identity + decile-average ordering."""
    # decode helper median ≡ Utils.median_forecast, exactly
    q = torch.arange(2 * 5 * 9, dtype=torch.float32).reshape(2, 5, 9)
    assert torch.equal(decode_point_forecast(q, "median"), Utils.median_forecast(q)), (
        "D3: decode_point_forecast('median') must equal Utils.median_forecast"
    )
    try:
        decode_point_forecast(q, "meanhead")
        raise AssertionError("D3: 'meanhead' must be rejected by the shared helper")
    except ValueError:
        pass
    # Decile average on lognormal quantiles: median < decile avg < exact mean
    from scipy import stats

    deciles = stats.lognorm.ppf(np.linspace(0.1, 0.9, 9), 1.0)
    median, decile_avg, exact_mean = deciles[4], float(np.mean(deciles)), float(np.exp(0.5))
    assert median < decile_avg < exact_mean, (
        f"D3: expected median {median:.3f} < decile-avg {decile_avg:.3f} < "
        f"exact mean {exact_mean:.3f} on lognormal(σ=1)"
    )
    print(
        f"✓ D3: helper median ≡ Utils.median_forecast; lognormal(σ=1) ordering "
        f"median {median:.3f} < decile-avg {decile_avg:.3f} < exact mean "
        f"{exact_mean:.3f} (decile average is biased LOW on right-skewed data)"
    )


def smoke_timesfm_layout(device: str, ctx: np.ndarray) -> None:
    """D2 — TimesFM output layout (go/no-go gate before the grid)."""
    import timesfm

    pipeline = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_SLUGS["timesfm"])
    pipeline.compile(
        timesfm.ForecastConfig(
            max_context=2048,
            per_core_batch_size=1,
            max_horizon=64,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    point, full = pipeline.forecast(inputs=[ctx], horizon=64)
    point = np.asarray(point).squeeze(0)
    full = np.asarray(full).squeeze(0)
    assert full.shape == (64, 10), f"D2: full forecast shape {full.shape} != (64, 10)"
    assert np.array_equal(point, full[:, 5]), (
        "D2 FAILED — STOP: TimesFM first return != full[:, 5]; the "
        "[mean, q0.1..q0.9] layout read is wrong, re-derive before the grid"
    )
    assert np.all(np.diff(full[:, 1:10], axis=1) >= 0), (
        "D2 FAILED — STOP: indices 1..9 not monotone, they are not the deciles"
    )
    assert not np.array_equal(full[:, 0], full[:, 5]), (
        "D2 FAILED — STOP: index 0 equals the median, no separate mean head"
    )
    delta = float(np.mean(full[:, 0] - full[:, 5]))
    decile_avg_delta = float(np.mean(full[:, 1:10].mean(axis=1) - full[:, 5]))
    print(
        f"✓ D2: TimesFM layout verified — first return ≡ full[:,5] (median); "
        f"deciles monotone; meanhead distinct. meanhead−median "
        f"{delta:+.4f} (expect >0 for right skew), decile-avg−median "
        f"{decile_avg_delta:+.4f} (normalized space, one trace)"
    )


def smoke_model(
    runner: CellRunner, model_name: str, args: argparse.Namespace
) -> None:
    """D1 (+D2 for TimesFM) + D4 for one model."""
    device = args.device
    slug = MODEL_SLUGS[model_name]
    print(f"\n--- Smoke: {slug} on {device} ---", flush=True)
    trace = runner.provider.get_id("iteration_8_ifft").numpy()
    ctx = (
        StandardScaler()
        .fit_transform(trace[:80].reshape(-1, 1))
        .squeeze()
        .astype(np.float32)
    )

    if model_name == "timesfm":
        smoke_timesfm_layout(device, ctx)

    # D1 — median path: in-process bit-reproducibility (licenses decoding
    # twins within one grid run) + the default factory call IS the median path
    predict_median = PREDICT_FACTORIES[model_name](device)
    out_a = np.asarray(predict_median(ctx, 64))
    out_b = np.asarray(predict_median(ctx, 64))
    assert np.array_equal(out_a, out_b), (
        f"D1: {model_name} median wrapper not bit-reproducible in-process"
    )

    if model_name == "tirex":
        # The library's "mean" return is a relabeled median — verify at runtime.
        import os

        os.environ.setdefault("TIREX_NO_CUDA", "1")
        from tirex import load_model

        pipeline = load_model(path=MODEL_SLUGS["tirex"], device=device)
        quantiles, lib_mean = pipeline.forecast(
            context=torch.tensor(ctx, dtype=torch.float32).to(device),
            prediction_length=64,
        )
        med = decode_point_forecast(quantiles, "median").squeeze().cpu().numpy()
        lib = np.asarray(lib_mean).squeeze()
        assert np.array_equal(med, lib), (
            "D1: TiRex library 'mean' return differs from the q0.5 median — "
            "the relabeled-median claim no longer holds, re-check tirex"
        )
        print("✓ D1+: TiRex library 'mean' return ≡ q0.5 median (relabel confirmed)")
        runner.cleanup(pipeline)

    # D1 — mean decoding differs from median; report the direction
    predict_mean = PREDICT_FACTORIES[model_name](device, "mean")
    out_mean = np.asarray(predict_mean(ctx, 64))
    assert out_mean.shape == out_a.shape and np.all(np.isfinite(out_mean))
    assert not np.array_equal(out_mean, out_a), (
        f"D1: {model_name} mean decode identical to median — knob inert?"
    )
    print(
        f"✓ D1: {model_name} median bit-reproducible; mean−median "
        f"{float(np.mean(out_mean - out_a)):+.4f} (normalized space, one pass)"
    )

    if model_name == "timesfm":
        predict_meanhead = PREDICT_FACTORIES[model_name](device, "meanhead")
        out_mh = np.asarray(predict_meanhead(ctx, 64))
        assert out_mh.shape == out_a.shape and np.all(np.isfinite(out_mh))
        assert not np.array_equal(out_mh, out_a), "D1: meanhead ≡ median?"
        print(
            f"✓ D1: timesfm meanhead wrapper finite, distinct from median "
            f"(meanhead−median {float(np.mean(out_mh - out_a)):+.4f})"
        )
        runner.cleanup(predict_meanhead)

    # D4 — timing: one mean-decoded k=10 full-trace rollout
    examples10 = runner.select_fns["ctx_euclid"](
        runner.pool, 10, 42, trace[:80], "iteration_8_ifft"
    )
    forecast_fn = make_concat_forecast_fn(predict_mean, "shared")
    config = FewShotConfig(
        device=device,
        model_slug=slug,
        model_prediction_length=64,
        start_context_length=80,
        relevant_prediction_tail=80,
        k_shot=10,
        random_seed=42,
        example_target_length=None,
        normalization="shared",
        point_stat="mean",
    )
    t0 = time.perf_counter()
    out = forecast_fn(trace, examples10, config)
    dt = time.perf_counter() - t0
    assert np.all(np.isfinite(out)), "D4: non-finite mean-decoded k=10 forecast"
    print(
        f"✓ D4: {model_name} mean k=10 single trace {dt:.1f}s -> det cell ≈ "
        f"{11 * dt / 60:.1f} min, {args.random_seeds}-seed random cell ≈ "
        f"{11 * args.random_seeds * dt / 60:.0f} min"
    )
    runner.cleanup(predict_median, predict_mean, forecast_fn)


def smoke(runner: CellRunner, args: argparse.Namespace) -> None:
    print(f"\n=== Smoke: decoding checks on {args.device} ===", flush=True)
    smoke_estimator_sanity()
    for model_name in args.models:
        smoke_model(runner, model_name, args)
    print("\n✅ Smoke test passed (no results written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-5 decoding grid")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="Which stages to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SLUGS),
        default=["tirex", "timesfm", "chronos2", "chronos_bolt"],
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=20,
        help="Number of seeds (0..n-1) for the random stage",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the D1-D4 sanity checks and exit (writes nothing)",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()

    runner = CellRunner(args)
    if args.smoke:
        smoke(runner, args)
        return

    t0 = time.perf_counter()
    for model_name in args.models:
        run_model_stages(runner, model_name, args)
    print(
        f"\n✅ Decoding grid stage(s) complete [{(time.perf_counter() - t0) / 60:.0f} min]",
        flush=True,
    )


if __name__ == "__main__":
    main()
