"""Re-run of the few-shot k-sweep with the FIXED example pool.

The published k-sweeps (results/few_shot_t80/ with 64-step example targets —
the README tables — and results/few_shot_t266/ with full-length example
targets) were run with an example pool that did not actually exclude the six
ID test traces (position-based instead of raw-id-based exclusion, fixed in
Phase 0). This runner repeats the exact old protocols — fixed pool, single
seed 42, k in {0, 1, 3, 5, 10} — so any number drift is attributable to the
pool fix alone.

Usage:
    # t266 protocol (full-length example targets)
    uv run python -m fusiontimeseries.benchmarking.few_shot.rerun_ksweep \
        --save-dir results/few_shot_v2_t266
    # t80 protocol (64-step example targets, the README tables)
    uv run python -m fusiontimeseries.benchmarking.few_shot.rerun_ksweep \
        --example-target-length 64 --save-dir results/few_shot_v2_t80
"""

import argparse
import gc
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    create_example_pool,
)
from fusiontimeseries.benchmarking.few_shot.harness import (
    PredictFn,
    make_icl_forecast_fn,
    run_benchmark,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    BenchmarkDataProvider,
    Utils,
)

MODEL_SLUGS: dict[str, str] = {
    "tirex": "NX-AI/TiRex",
    "chronos2": "amazon/chronos-2",
    "chronos_bolt": "amazon/chronos-bolt-tiny",
    "timesfm": "google/timesfm-2.5-200m-pytorch",
}


def make_tirex_predict(device: str) -> PredictFn:
    """TiRex: 1D device-placed input, quantile output [1, pred_len, 9]."""
    import os

    if not device.startswith("cuda"):
        # Disable the sLSTM CUDA kernels (JIT-compiled, CUDA-only) so TiRex
        # falls back to the vanilla path on MPS/CPU. Must be set before the
        # first forward pass.
        os.environ.setdefault("TIREX_NO_CUDA", "1")
    from tirex import load_model

    pipeline = load_model(path=MODEL_SLUGS["tirex"], device=device)

    def predict(context: NDArray[np.float32], prediction_length: int) -> NDArray[np.float32]:
        ctx_tensor = torch.tensor(context, dtype=torch.float32)
        quantiles, _ = pipeline.forecast(
            context=ctx_tensor.to(device), prediction_length=prediction_length
        )
        return Utils.median_forecast(quantiles).squeeze().cpu().numpy()

    return predict


def make_chronos2_pipeline(device: str):
    """Load the Chronos-2 pipeline (shared by concat and group-ICL wrappers)."""
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(
        pretrained_model_name_or_path=MODEL_SLUGS["chronos2"],
        device_map=device,
        dtype=torch.bfloat16,
    )


def chronos2_predict_from_pipeline(pipeline) -> PredictFn:
    """Wrap a loaded Chronos-2 pipeline into the standard PredictFn."""

    def predict(context: NDArray[np.float32], prediction_length: int) -> NDArray[np.float32]:
        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        forecast = pipeline.predict(inputs=ctx_tensor, prediction_length=prediction_length)
        quantiles = forecast[0].permute(0, 2, 1)  # [1, pred_len, n_quantiles]
        return Utils.median_forecast(quantiles).squeeze().cpu().numpy()

    return predict


def make_chronos2_predict(device: str) -> PredictFn:
    """Chronos-2: CPU input [1, 1, ctx] (handles device internally)."""
    return chronos2_predict_from_pipeline(make_chronos2_pipeline(device))


def make_chronos_bolt_predict(device: str) -> PredictFn:
    """Chronos-Bolt: device-placed input [1, ctx], quantile-first output."""
    from chronos import ChronosBoltPipeline

    pipeline = ChronosBoltPipeline.from_pretrained(
        pretrained_model_name_or_path=MODEL_SLUGS["chronos_bolt"],
        device_map=device,
        dtype=torch.bfloat16,
    )

    def predict(context: NDArray[np.float32], prediction_length: int) -> NDArray[np.float32]:
        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        quantiles = pipeline.predict(
            inputs=ctx_tensor.to(device), prediction_length=prediction_length
        ).permute(0, 2, 1)  # [1, pred_len, n_quantiles]
        return Utils.median_forecast(quantiles).squeeze().cpu().numpy()

    return predict


def make_timesfm_predict(device: str) -> PredictFn:
    """TimesFM 2.5: list-of-arrays input, point forecast output."""
    import timesfm

    pipeline = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_SLUGS["timesfm"])
    pipeline.compile(
        timesfm.ForecastConfig(
            max_context=2048,  # matches the original few-shot runs
            per_core_batch_size=1,
            max_horizon=64,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    def predict(context: NDArray[np.float32], prediction_length: int) -> NDArray[np.float32]:
        forecast, _ = pipeline.forecast(inputs=[context], horizon=prediction_length)
        return np.asarray(forecast).squeeze(0)

    return predict


PREDICT_FACTORIES = {
    "tirex": make_tirex_predict,
    "chronos2": make_chronos2_predict,
    "chronos_bolt": make_chronos_bolt_predict,
    "timesfm": make_timesfm_predict,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot k-sweep with the fixed pool")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SLUGS),
        default=["tirex", "timesfm", "chronos2", "chronos_bolt"],
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[0, 1, 3, 5, 10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--example-target-length",
        type=lambda s: None if s.lower() == "none" else int(s),
        default=None,
        help="Example target length: 64 for the t80 protocol, 'none' for t266 (default)",
    )
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args()

    pool = create_example_pool(
        exclude_ids=set(ID_TEST_RAW_IDS), target_length=args.example_target_length
    )
    assert len(pool) == 245, f"Expected fixed pool of 245, got {len(pool)}"
    provider = BenchmarkDataProvider()

    for model_name in args.models:
        slug = MODEL_SLUGS[model_name]
        model_name_clean = slug.replace("/", "_")
        print(f"\n{'=' * 60}\nLoading {slug} on {args.device}\n{'=' * 60}", flush=True)
        predict_fn = PREDICT_FACTORIES[model_name](args.device)
        forecast_fn = make_icl_forecast_fn(predict_fn)

        for k in args.ks:
            config = FewShotConfig(
                device=args.device,
                model_slug=slug,
                model_prediction_length=64,
                start_context_length=80,
                relevant_prediction_tail=80,
                k_shot=k,
                random_seed=args.seed,
                example_target_length=args.example_target_length,
            )
            results = run_benchmark(
                forecast_fn=forecast_fn,
                config=config,
                example_pool=pool,
                method=model_name_clean,
                seeds=(args.seed,),
                provider=provider,
                save_dir=args.save_dir,
            )
            print(
                f"[{model_name}] k={k}: "
                f"ID {results.in_distribution.rmse:.2f} ± {results.in_distribution.se_rmse:.2f}, "
                f"OOD {results.out_of_distribution.rmse:.2f} ± {results.out_of_distribution.se_rmse:.2f}",
                flush=True,
            )

        del predict_fn, forecast_fn
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    print("\n✅ k-sweep complete", flush=True)


if __name__ == "__main__":
    main()
