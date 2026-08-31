"""Calibrate the point-forecast decoding quantile on TRAINING traces, not the benchmark.

The 2026-06-14 quantile sweep (``explore_decoding_sweep.py``) found that a
quantile ABOVE the median beats median/quantile-mean for every model — but it
selected that quantile by argmin over the 6 ID / 5 OOD *benchmark* traces, i.e.
by reading the test labels. ``docs/methods/decoding_and_ensembling.md`` flags
this: "still tuned on n=6/5, so it needs validation on held-out traces before it
is claimed as a method." This module supplies that validation.

Design
------
**Calibration set** = the 244 non-benchmark pool traces that carry operating
parameters (``create_example_pool(exclude_ids=ID_TEST_RAW_IDS)`` minus raw 300,
which has no dump entry). Each is scored under the *identical* protocol the
benchmark uses: 80-step context, autoregressive rollout to the end of the
267-step trace, tail metric = ``mean(x[-80:])``. Retrieval is
**leave-one-out** — the query's own raw id is removed from its example pool.

**Leakage status per model.** The four base TSFMs never saw any GyroKinetic
trace, so all 244 are genuinely held out *from the model* and this is a clean
calibration split. The finetuned Chronos-2 is the opposite case: 241 of the 244
are in ``TRAIN_IDXS``, so its calibration is CONTAMINATED — reported anyway,
with the 3 traces outside ``TRAIN_IDXS`` (13/100/200, the finetuning validation
set) tagged so the analyzer can score them separately. The bias has a known
direction: the model under-predicts less on traces it fit, so a
train-calibrated quantile is a LOWER BOUND on the shift a held-out query needs.
Removing that bias requires k-fold *retraining*, not k-fold over traces.

**Benchmark set** = the same 11 traces, same grid, so the analyzer can run the
transfer test (calibration-selected q vs. median vs. mean vs. the test-selected
"oracle q") on identical footing. This also produces the ft model's test-side
quantile grid, which the base-only 06-14 sweep never covered.

Output is one row per (model, evalset, config, spec, trace) written to a
resumable JSONL; every downstream question — argmin, k-fold, transfer — is
answered offline by ``analyze_decoding_calibration.py`` from those rows, so no
analysis choice costs another rollout.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.calibrate_decoding \
        --device mps --models chronos_bolt chronos2 ft_chronos2 timesfm tirex
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    FewShotExample,
    create_example_pool,
)
from fusiontimeseries.benchmarking.few_shot.harness import (
    IN_DISTRIBUTION_ITERATIONS,
    OUT_OF_DISTRIBUTION_ITERATIONS,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    ID_TEST_RAW_IDS,
    get_params_for_benchmark_trace,
)
from fusiontimeseries.benchmarking.few_shot.presentation import make_concat_forecast_fn
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    MODEL_SLUGS,
    make_chronos2_pipeline,
)
from fusiontimeseries.benchmarking.few_shot.selection import make_select_fn
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import BenchmarkDataProvider
from fusiontimeseries.lib.dataset import TRAIN_IDXS

#: Quantile levels to probe (resolved to each model's nearest available level).
REQUESTED_LEVELS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
#: (strategy, k) — all deterministic, single seed 42. The first two are the
#: 06-14 sweep's configs; ``ctx_level`` was added 2026-08-30 because it carries
#: the report's OOD numbers and had no off-benchmark decoding of its own.
CONFIGS: tuple[tuple[str, int], ...] = (
    ("zeroshot", 0),
    ("mmr_euclid", 5),
    ("ctx_level", 5),
)
#: Report checkpoint behind the ft rows (Phase-6 v6 grid / the 15.63 ID cell).
FT_CHECKPOINT = Path("results/few_shot_v6_finetuned/checkpoint/lora_weights.pt")

DEFAULT_OUT = Path("results/few_shot_v8_calibration/decoding_calibration.jsonl")

MODELS: tuple[str, ...] = ("chronos_bolt", "chronos2", "ft_chronos2", "timesfm", "tirex")


def _nearest_idx(levels: list[float], target: float) -> int:
    return int(np.argmin(np.abs(np.array(levels) - target)))


def _label(spec) -> str:
    return spec if isinstance(spec, str) else f"q{spec:.2f}"


########################################################
# Model builders: predict_for(spec, params) -> PredictFn
########################################################


def build(model: str, device: str):
    """Load one model ONCE. Returns (obj, levels, predict_for, has_meanhead).

    ``predict_for(spec, params)`` builds a PredictFn for spec ∈ {float level,
    "mean", "meanhead"}. ``params`` is the query trace's RAW operating-param
    dict; only the finetuned model uses it (its BilinearLoRA forward raises
    without a ``ConditionRegistry`` entry), the base builders ignore it.
    """
    if model == "chronos2":
        pipe = make_chronos2_pipeline(device)
        levels = [float(q) for q in pipe.model.chronos_config.quantiles]  # 21

        def predict_for(spec, params=None):
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

    if model == "ft_chronos2":
        from fusiontimeseries.benchmarking.few_shot.finetuned import (
            FT_TRAIN_CONTEXT,
            load_finetuned_chronos2,
            raw_param_tensor,
        )
        from fusiontimeseries.lib.conditioning import ConditionRegistry

        pipe = load_finetuned_chronos2(
            FT_CHECKPOINT, device, context_window=FT_TRAIN_CONTEXT
        )
        levels = [float(q) for q in pipe.model.chronos_config.quantiles]  # 21

        def predict_for(spec, params=None):
            if params is None:
                raise ValueError("ft_chronos2 requires the query's operating params")
            op_params = raw_param_tensor(params)

            def predict(context, h):
                with ConditionRegistry.patch(op_params=op_params):
                    f = pipe.predict(
                        inputs=torch.tensor(context, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0),
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
        levels = [float(q) for q in pipe.quantiles]  # 9 @ 0.1..0.9

        def predict_for(spec, params=None):
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
        levels = [round(0.1 * i, 1) for i in range(1, 10)]

        def predict_for(spec, params=None):
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

        def predict_for(spec, params=None):
            def predict(context, h):
                _, full = pipe.forecast(inputs=[context], horizon=h)
                full = np.asarray(full).squeeze(0)  # [h, 10]
                if spec == "mean":
                    return full[:, 1:10].mean(axis=-1)
                if spec == "meanhead":
                    return full[:, 0]
                return full[:, _nearest_idx(levels, spec) + 1]
            return predict

        return pipe, levels, predict_for, True

    raise ValueError(f"Unknown model {model!r}")


########################################################
# Evaluation sets
########################################################


class Query:
    """One evaluation query: its trace, its retrieval pool, its OP params."""

    __slots__ = ("key", "evalset", "split", "trace", "pool", "params", "in_ft_train")

    def __init__(self, key, evalset, split, trace, pool, params, in_ft_train):
        self.key = key
        self.evalset = evalset
        self.split = split
        self.trace = trace
        self.pool = pool
        self.params = params
        self.in_ft_train = in_ft_train


def build_queries(full_pool: list[FewShotExample]) -> list[Query]:
    """Calibration queries (LOO pool) followed by the 11 benchmark queries."""
    queries: list[Query] = []

    # --- calibration: every pool trace with operating params, LOO retrieval ---
    for ex in full_pool:
        if ex.operating_params is None:  # raw 300: no dump entry
            continue
        loo_pool = [o for o in full_pool if o.trace_id != ex.trace_id]
        queries.append(
            Query(
                key=f"cal_{ex.trace_id}",
                evalset="calibration",
                split="cal",
                trace=ex.trace_array,
                pool=loo_pool,
                params=ex.operating_params,
                in_ft_train=ex.trace_id in TRAIN_IDXS,
            )
        )

    # --- benchmark: the shipped 11, full pool (the shipped protocol) ---
    provider = BenchmarkDataProvider()
    for split, keys, getter in (
        ("id", IN_DISTRIBUTION_ITERATIONS, provider.get_id),
        ("ood", OUT_OF_DISTRIBUTION_ITERATIONS, provider.get_ood),
    ):
        for key in keys:
            queries.append(
                Query(
                    key=key,
                    evalset="benchmark",
                    split=split,
                    trace=getter(key).numpy(),
                    pool=full_pool,
                    params=get_params_for_benchmark_trace(key),
                    in_ft_train=False,
                )
            )
    return queries


########################################################
# Runner
########################################################


def load_done(out_path: Path) -> set[tuple[str, str, str, int, str]]:
    """(model, evalset, strategy, k, spec) groups already complete on disk."""
    if not out_path.exists():
        return set()
    counts: dict[tuple, int] = {}
    with out_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["model"], r["evalset"], r["strategy"], r["k"], r["spec"])
            counts[key] = counts.get(key, 0) + 1
    return set(counts)  # group presence; expected-count check happens in main


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-side decoding calibration sweep")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-calibration", type=int, default=None,
        help="cap the calibration set (stratified by tail level) — for smoke runs",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    full_pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    queries = build_queries(full_pool)

    if args.max_calibration is not None:
        cal = [q for q in queries if q.evalset == "calibration"]
        bench = [q for q in queries if q.evalset == "benchmark"]
        order = np.argsort([float(np.mean(q.trace[-80:])) for q in cal])
        pick = sorted(order[:: max(1, len(cal) // args.max_calibration)][: args.max_calibration])
        queries = [cal[i] for i in pick] + bench

    n_cal = sum(1 for q in queries if q.evalset == "calibration")
    print(f"queries: {n_cal} calibration + {len(queries) - n_cal} benchmark", flush=True)

    select_fns = {
        s: make_select_fn(s)
        for s in {strategy for strategy, k in CONFIGS if k > 0}
    }
    # Retrieved examples depend only on (query, k) — hoisted out of the spec loop.
    example_cache: dict[tuple[str, str, int], list[FewShotExample]] = {}

    def examples_for(q: Query, strategy: str, k: int) -> list[FewShotExample]:
        if k == 0:
            return []
        ck = (q.key, strategy, k)
        if ck not in example_cache:
            example_cache[ck] = select_fns[strategy](
                q.pool, k, 42, q.trace[:80], q.key
            )
        return example_cache[ck]

    done_groups = load_done(args.out)
    fh = args.out.open("a")

    for model in args.models:
        print(f"\n{'=' * 78}\n{model} on {args.device}\n{'=' * 78}", flush=True)
        t_load = time.time()
        obj, levels, predict_for, has_meanhead = build(model, args.device)
        print(f"  loaded in {time.time() - t_load:.1f}s; grid ({len(levels)}): {levels}", flush=True)

        resolved = sorted({levels[_nearest_idx(levels, t)] for t in REQUESTED_LEVELS})
        specs: list = list(resolved) + ["mean"] + (["meanhead"] if has_meanhead else [])
        needs_params = model == "ft_chronos2"

        for strategy, k in CONFIGS:
            for spec in specs:
                gkey = (model, "calibration", strategy, k, _label(spec))
                bkey = (model, "benchmark", strategy, k, _label(spec))
                todo = [
                    q for q in queries
                    if (gkey if q.evalset == "calibration" else bkey) not in done_groups
                ]
                if not todo:
                    print(f"  {strategy} k={k} {_label(spec):9} — cached, skipped", flush=True)
                    continue

                t0 = time.time()
                # Base models: one forecast_fn for the whole spec (params unused).
                shared_fc = (
                    None if needs_params
                    else make_concat_forecast_fn(predict_for(spec), "shared")
                )
                cfg = FewShotConfig(
                    device=args.device, model_slug=MODEL_SLUGS.get(model, model),
                    model_prediction_length=64, start_context_length=80,
                    relevant_prediction_tail=80, k_shot=k, random_seed=42,
                    example_target_length=None, normalization="shared",
                )
                for q in todo:
                    fc = (
                        make_concat_forecast_fn(predict_for(spec, q.params), "shared")
                        if needs_params else shared_fc
                    )
                    forecast = fc(q.trace, examples_for(q, strategy, k), cfg)
                    fh.write(json.dumps({
                        "model": model, "evalset": q.evalset, "split": q.split,
                        "trace": q.key, "strategy": strategy, "k": k,
                        "spec": _label(spec), "in_ft_train": q.in_ft_train,
                        "true_tail": float(np.mean(q.trace[-80:])),
                        "pred_tail": float(np.mean(forecast[-80:])),
                    }) + "\n")
                fh.flush()
                dt = time.time() - t0
                print(
                    f"  {strategy} k={k} {_label(spec):9} — {len(todo)} traces "
                    f"in {dt / 60:.1f} min ({dt / max(1, len(todo)):.2f} s/trace)",
                    flush=True,
                )

        del obj, predict_for
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    fh.close()
    print(f"\n✅ wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
