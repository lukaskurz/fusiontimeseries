"""Exploration: is a fixed quantile LEVEL better than mean/median decoding? (Phase-5 follow-up)

Phase 5 only compared median vs the decile-mean. This sweeps the point-forecast
decoding over a range of quantile LEVELS (0.5 .. 0.99), the decile/quantile
mean, and (TimesFM) the native mean head, at the two most informative configs —
zero-shot k=0 (most decoding-sensitive: the per-step distribution is widest) and
the best-legit retrieval config (mmr_euclid shared k=5) — for all four models.

It is LEVEL-based, not index-based, because the models do NOT share a quantile
grid: Chronos-2 emits 21 quantiles (0.01..0.99), while Chronos-Bolt / TiRex /
TimesFM emit 9 (0.1..0.9). A requested level is resolved to each model's nearest
available quantile and the resolved level is reported.

Questions:
  1. Is there a FIXED quantile level better calibrated than the median (0.5) or
     the quantile-mean (the two Phase-5 options)?
  2. Is that best level CONSISTENT across the two configs (a robust global
     calibration shift worth adopting) or config-dependent (overfitting to the
     6 ID / 5 OOD test traces — a mild oracle)?

CAVEAT: picking "the level that minimizes test RMSE" reads the test labels, so a
per-config best level is a CALIBRATION CEILING / diagnostic, not a deployable
method (like oracle_tail). Only a level that wins CONSISTENTLY across configs is
a legitimate recommendation. n=6 ID / 5 OOD.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.explore_decoding_sweep \
        --device mps --models chronos2 chronos_bolt tirex timesfm
"""

import argparse
import gc

import numpy as np
import torch

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    create_example_pool,
)
from fusiontimeseries.benchmarking.few_shot.harness import run_benchmark
from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
from fusiontimeseries.benchmarking.few_shot.presentation import make_concat_forecast_fn
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    MODEL_SLUGS,
    make_chronos2_pipeline,
)
from fusiontimeseries.benchmarking.few_shot.selection import make_select_fn
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import BenchmarkDataProvider

#: Quantile levels to probe (resolved to each model's nearest available level).
REQUESTED_LEVELS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
#: (strategy, k) — both deterministic, single seed 42.
CONFIGS: tuple[tuple[str, int], ...] = (("zeroshot", 0), ("mmr_euclid", 5))


def _nearest_idx(levels: list[float], target: float) -> int:
    return int(np.argmin(np.abs(np.array(levels) - target)))


def build(model: str, device: str):
    """Load one model ONCE. Returns (obj, levels, predict_for, has_meanhead).

    ``predict_for(spec)`` builds a PredictFn for spec ∈ {float level, "mean",
    "meanhead"}, decoding the raw quantile tensor at the level's nearest index.
    """
    if model == "chronos2":
        pipe = make_chronos2_pipeline(device)
        levels = [float(q) for q in pipe.model.chronos_config.quantiles]  # 21

        def predict_for(spec):
            def predict(context, h):
                f = pipe.predict(
                    inputs=torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    prediction_length=h,
                )
                raw = f[0].permute(0, 2, 1)  # [1, h, 21]
                if spec == "mean":
                    return raw.mean(dim=-1).squeeze().cpu().numpy()
                return raw[..., _nearest_idx(levels, spec)].squeeze().cpu().numpy()
            return predict

        return pipe, levels, predict_for, False

    if model == "chronos_bolt":
        from chronos import ChronosBoltPipeline

        pipe = ChronosBoltPipeline.from_pretrained(
            pretrained_model_name_or_path=MODEL_SLUGS["chronos_bolt"],
            device_map=device, dtype=torch.bfloat16,
        )
        levels = [float(q) for q in pipe.quantiles]  # [0.1..0.9]

        def predict_for(spec):
            def predict(context, h):
                raw = pipe.predict(
                    inputs=torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device),
                    prediction_length=h,
                ).permute(0, 2, 1)  # [1, h, 9]
                if spec == "mean":
                    return raw.mean(dim=-1).squeeze().cpu().numpy()
                return raw[..., _nearest_idx(levels, spec)].squeeze().cpu().numpy()
            return predict

        return pipe, levels, predict_for, False

    if model == "tirex":
        import os

        if not device.startswith("cuda"):
            os.environ.setdefault("TIREX_NO_CUDA", "1")
        from tirex import load_model

        pipe = load_model(path=MODEL_SLUGS["tirex"], device=device)
        levels = [round(0.1 * i, 1) for i in range(1, 10)]  # 9 deciles 0.1..0.9

        def predict_for(spec):
            def predict(context, h):
                raw, _ = pipe.forecast(
                    context=torch.tensor(context, dtype=torch.float32).to(device),
                    prediction_length=h,
                )  # [1, h, 9]
                if spec == "mean":
                    return raw.mean(dim=-1).squeeze().cpu().numpy()
                return raw[..., _nearest_idx(levels, spec)].squeeze().cpu().numpy()
            return predict

        return pipe, levels, predict_for, False

    if model == "timesfm":
        import timesfm

        pipe = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_SLUGS["timesfm"])
        pipe.compile(
            timesfm.ForecastConfig(
                max_context=2048, per_core_batch_size=1, max_horizon=64,
                normalize_inputs=True, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        levels = [round(0.1 * i, 1) for i in range(1, 10)]  # cols 1..9 of [mean,q0.1..q0.9]

        def predict_for(spec):
            def predict(context, h):
                _, full = pipe.forecast(inputs=[context], horizon=h)
                full = np.asarray(full).squeeze(0)  # [h, 10] = [mean, q0.1..q0.9]
                if spec == "mean":
                    return full[:, 1:10].mean(axis=-1)
                if spec == "meanhead":
                    return full[:, 0]
                return full[:, _nearest_idx(levels, spec) + 1]
            return predict

        return pipe, levels, predict_for, True

    raise ValueError(f"Unknown model {model!r}")


def _label(spec) -> str:
    return spec if isinstance(spec, str) else f"q{spec:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantile-level decoding sweep (Phase-5 follow-up)")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--models", nargs="+", default=["chronos2", "chronos_bolt", "tirex", "timesfm"],
        choices=["chronos2", "chronos_bolt", "tirex", "timesfm"],
    )
    args = parser.parse_args()

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    provider = BenchmarkDataProvider()
    mmr = make_select_fn("mmr_euclid")

    for model in args.models:
        slug = MODEL_SLUGS[model]
        print(f"\n{'=' * 80}\n{model} ({slug}) on {args.device}\n{'=' * 80}", flush=True)
        obj, levels, predict_for, has_meanhead = build(model, args.device)
        # Resolve requested levels to this model's grid; dedupe; add mean(+meanhead).
        resolved = sorted({levels[_nearest_idx(levels, t)] for t in REQUESTED_LEVELS})
        specs: list = list(resolved) + ["mean"] + (["meanhead"] if has_meanhead else [])
        print(f"  quantile grid ({len(levels)}): {levels}", flush=True)

        rows: dict[object, dict[tuple[str, int], tuple[float, float]]] = {}
        for spec in specs:
            fc = make_concat_forecast_fn(predict_for(spec), "shared")
            rows[spec] = {}
            for strategy, k in CONFIGS:
                config = FewShotConfig(
                    device=args.device, model_slug=slug, model_prediction_length=64,
                    start_context_length=80, relevant_prediction_tail=80, k_shot=k,
                    random_seed=42, example_target_length=None, normalization="shared",
                    point_stat=spec if spec in ("mean", "meanhead") else "median",
                )
                select = None if strategy == "zeroshot" else mmr
                res = run_benchmark(
                    forecast_fn=fc, config=config, example_pool=pool,
                    method=f"sweep_{slug.replace('/', '_')}_{strategy}_{_label(spec)}",
                    select_fn=select, seeds=(42,), deterministic=True,
                    provider=provider, save=False,
                )
                rows[spec][(strategy, k)] = (res.in_distribution.rmse, res.out_of_distribution.rmse)
            print(
                f"  {_label(spec):9} | "
                + " | ".join(
                    f"{s} k={k}: {rows[spec][(s, k)][0]:6.2f} / {rows[spec][(s, k)][1]:6.2f}"
                    for s, k in CONFIGS
                ),
                flush=True,
            )

        median_spec = 0.5 if 0.5 in resolved else resolved[_nearest_idx(resolved, 0.5)]
        print(f"\n  --- {model}: best decoding per column (ID / OOD tail RMSE; median=q0.50, mean=quantile avg) ---")
        for strategy, k in CONFIGS:
            for si, split in enumerate(("ID", "OOD")):
                best = min(specs, key=lambda s: rows[s][(strategy, k)][si])
                med = rows[median_spec][(strategy, k)][si]
                mean = rows["mean"][(strategy, k)][si]
                print(
                    f"    {strategy} k={k} {split}: best={_label(best)} "
                    f"({rows[best][(strategy, k)][si]:.2f}) vs median {med:.2f} / mean {mean:.2f}"
                )
        zs_best = _label(min(specs, key=lambda s: rows[s][("zeroshot", 0)][0]))
        mmr_best = _label(min(specs, key=lambda s: rows[s][("mmr_euclid", 5)][0]))
        verdict = "CONSISTENT (robust)" if zs_best == mmr_best else "config-dependent (overfit risk)"
        print(f"    ID-best: zeroshot={zs_best}, mmr_k5={mmr_best} -> {verdict}")

        del obj, predict_for
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    print("\n✅ Decoding sweep complete (no results written)", flush=True)


if __name__ == "__main__":
    main()
