"""Phase-4 covariate-conditioning grid: op-param channels x strategies x k.

Chronos-2 ONLY (the only benchmarked model with zero-shot covariate
support); every cell uses Phase-3's winning presentation (flat concat +
SHARED scaling), t266 protocol, fixed 245-trace pool. All cells write to ONE
save dir (``results/few_shot_v4_covariates/``) so every headline comparison
lives within a single grid run. Stages:

- ``anchors``: zero-shot k=0 without and WITH the (constant, hence
  structurally inert) covariate channels — the empirical inertness row and
  the row-presence offset.
- ``main``:  {random(20 seeds), op_knn, ctx_euclid, mmr_euclid, oracle_tail}
  x k{3,5,10} x {no-cov, +cov}.
- ``k1``:    {ctx_euclid, op_knn} x k=1 x {no-cov, +cov} — the sign-bit
  degeneracy diagnostic (a k=1 step channel carries one bit per param).
- ``perm``:  {random(20), ctx_euclid} x k{5,10} with PERMUTED example params
  (query true) — the attribution control: +cov gains that don't separate
  from permcov are "extra rows present", not parameter information.
- ``group``: {random(5 seeds), ctx_euclid} x k{5,10} x {group no-cov,
  group+cov} — group mode has no per-example param slot and constant
  channels are erased by the per-row norm; this block puts that structural
  inertness claim in the table.

= 2 + 30 + 4 + 4 + 8 = 48 cells. No-cov cells run through the frozen
``presentation.make_concat_forecast_fn`` path — licensed by smoke S2, which
asserts the dict-task-no-cov path is equivalent to the plain-tensor path.

Method labels: ``amazon_chronos-2_{strategy}__{variant}`` with variant
tokens ``shared`` | ``shared-opcov`` | ``shared-permcov`` | ``group`` |
``group-opcov``. A cell whose result file already exists in the save dir is
skipped (resumable).

Usage (single staged background run):
    uv run python -m fusiontimeseries.benchmarking.few_shot.run_covariates_grid \
        --device mps
    ... --stages main perm        # subset of stages
    ... --smoke                   # S1-S5 sanity checks, writes nothing

``--smoke`` loads the real Chronos-2 and runs: S1a structural inertness
(zeroshot+cov params A vs B), S1b row-presence offset (zeroshot+cov vs
zeroshot), S1c group inertness, S2 path equivalence (dict-no-cov vs plain
tensor), S3 sensitivity (true vs opposite-range query params at k=5),
S4 NaN mid-stream (the params-less pool example), S5 timing calibration.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from fusiontimeseries.benchmarking.few_shot.covariates import (
    make_chronos2_covariate_forecast_fn,
    make_chronos2_group_covariate_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotConfig,
    FewShotExample,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    OP_NAMES,
    get_params_for_benchmark_trace,
)
from fusiontimeseries.benchmarking.few_shot.presentation import (
    make_chronos2_group_forecast_fn,
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

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_SAVE_DIR: Path = REPO_ROOT / "results" / "few_shot_v4_covariates"

STAGES: tuple[str, ...] = ("anchors", "main", "k1", "perm", "group")
STAGE_KS: dict[str, list[int]] = {
    "main": [3, 5, 10],
    "k1": [1],
    "perm": [5, 10],
    "group": [5, 10],
}
MAIN_STRATEGIES: tuple[str, ...] = (
    "random",
    "op_knn",
    "ctx_euclid",
    "mmr_euclid",
    "oracle_tail",
)
K1_STRATEGIES: tuple[str, ...] = ("ctx_euclid", "op_knn")
PERM_STRATEGIES: tuple[str, ...] = ("random", "ctx_euclid")
GROUP_STRATEGIES: tuple[str, ...] = ("random", "ctx_euclid")

CHRONOS2_SLUG: str = MODEL_SLUGS["chronos2"]


def opposite_range_params(params: dict[str, float]) -> dict[str, float]:
    """Reflect each operating parameter across its FTSConfig range midpoint."""
    from fusiontimeseries.lib.config import FTSConfig

    op_ranges = FTSConfig().op_ranges
    return {name: op_ranges[name][0] + op_ranges[name][1] - params[name] for name in OP_NAMES}


########################################################
# Stages
########################################################


def stage_anchors(runner: CellRunner, fns: dict, args: argparse.Namespace) -> None:
    """Zero-shot k=0 anchors: no-cov and +cov (the empirical inertness row)."""
    print(f"\n=== Stage anchors: {CHRONOS2_SLUG} on {args.device} ===", flush=True)
    runner.run_cell(
        fns["shared"], CHRONOS2_SLUG, "zeroshot", 0, None, (42,), True,
        normalization="shared",
    )
    runner.run_cell(
        fns["cov"], CHRONOS2_SLUG, "zeroshot", 0, None, (42,), True,
        normalization="shared", op_covariates="step",
    )


def stage_main(runner: CellRunner, fns: dict, args: argparse.Namespace) -> None:
    """Main grid: 5 strategies x k{3,5,10} x {no-cov, +cov}."""
    ks = args.ks or STAGE_KS["main"]
    strategies = args.strategies or MAIN_STRATEGIES
    print(
        f"\n=== Stage main: {CHRONOS2_SLUG} on {args.device}, ks={ks}, "
        f"strategies={list(strategies)} ===",
        flush=True,
    )
    for strategy in strategies:
        seeds, deterministic = seeds_for(strategy, args.random_seeds)
        for k in ks:
            # Same select_fn + same seeds => identical example sets across
            # the cov twin (the analyzer hard-asserts this).
            for forecast_fn, op_covariates in ((fns["shared"], None), (fns["cov"], "step")):
                runner.run_cell(
                    forecast_fn, CHRONOS2_SLUG, strategy, k,
                    runner.select_fns[strategy], seeds, deterministic,
                    normalization="shared", op_covariates=op_covariates,
                )


def stage_k1(runner: CellRunner, fns: dict, args: argparse.Namespace) -> None:
    """k=1 diagnostic: the step channel degenerates to one sign bit per param."""
    print(f"\n=== Stage k1: {CHRONOS2_SLUG} on {args.device} ===", flush=True)
    for strategy in K1_STRATEGIES:
        seeds, deterministic = seeds_for(strategy, args.random_seeds)
        for forecast_fn, op_covariates in ((fns["shared"], None), (fns["cov"], "step")):
            runner.run_cell(
                forecast_fn, CHRONOS2_SLUG, strategy, 1,
                runner.select_fns[strategy], seeds, deterministic,
                normalization="shared", op_covariates=op_covariates,
            )


def stage_perm(runner: CellRunner, fns: dict, args: argparse.Namespace) -> None:
    """Permuted-params control: example params shuffled, query true."""
    ks = args.ks or STAGE_KS["perm"]
    print(f"\n=== Stage perm: {CHRONOS2_SLUG} on {args.device}, ks={ks} ===", flush=True)
    for strategy in PERM_STRATEGIES:
        seeds, deterministic = seeds_for(strategy, args.random_seeds)
        for k in ks:
            runner.run_cell(
                fns["perm"], CHRONOS2_SLUG, strategy, k,
                runner.select_fns[strategy], seeds, deterministic,
                normalization="shared", op_covariates="permuted",
            )


def stage_group(runner: CellRunner, fns: dict, args: argparse.Namespace) -> None:
    """Group block: per-task constant channels are structurally inert."""
    ks = args.ks or STAGE_KS["group"]
    print(f"\n=== Stage group: {CHRONOS2_SLUG} on {args.device}, ks={ks} ===", flush=True)
    for strategy in GROUP_STRATEGIES:
        seeds, deterministic = seeds_for(strategy, args.group_random_seeds)
        for k in ks:
            for forecast_fn, op_covariates in (
                (fns["group"], None),
                (fns["group_cov"], "step"),
            ):
                runner.run_cell(
                    forecast_fn, CHRONOS2_SLUG, strategy, k,
                    runner.select_fns[strategy], seeds, deterministic,
                    presentation="group", op_covariates=op_covariates,
                )


########################################################
# Smoke test (real Chronos-2, no results written)
########################################################


def smoke(runner: CellRunner, pipeline, fns: dict, args: argparse.Namespace) -> None:
    print(f"\n=== Smoke: real Chronos-2 covariates on {args.device} ===", flush=True)
    trace = runner.provider.get_id("iteration_8_ifft").numpy()
    true_params = get_params_for_benchmark_trace("iteration_8_ifft")
    opposite = opposite_range_params(true_params)
    print(f"  query params: {true_params}")
    print(f"  opposite-range params: {opposite}")

    def make_config(k: int) -> FewShotConfig:
        return FewShotConfig(
            device=args.device,
            model_slug=CHRONOS2_SLUG,
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=k,
            random_seed=42,
            example_target_length=None,
            normalization="shared",
        )

    def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    # A constant channel row is instance-normed to a constant all-0/±arcsinh(1)
    # row — the float32 rounding direction of the row mean (covariates.py C5).
    # The parameter VALUE is erased, but the surviving tri-state bits still
    # enter attention, so two param sets with DIFFERENT tri-states give
    # genuinely different model inputs. S1 therefore separates three
    # measurements: reproducibility, VALUE-erasure (tri-state-matched params
    # -> identical outputs), and the artifact magnitude (tri-states differ ->
    # reported). The tri-state is 1-ulp rounding of a float32 mean and hence
    # DEVICE-dependent (CPU and MPS reduce in different orders): probe it
    # through the pipeline model's own InstanceNorm on the model's device, in
    # the same 5-row shape the real forward sees.
    model = pipeline.model if hasattr(pipeline, "model") else pipeline.inner_model
    norm_layer = model.instance_norm
    model_device = next(model.parameters()).device

    def tri_states_on_device(values: np.ndarray, length: int) -> np.ndarray:
        """Post-norm constant of each candidate's (1+4)-row-shaped const row."""
        states = np.empty(len(values), dtype=np.float64)
        filler = np.linspace(-1.0, 1.0, length, dtype=np.float32)  # dummy target row
        for start in range(0, len(values), 4):
            chunk = values[start : start + 4]
            rows = np.concatenate(
                [filler[None, :], np.repeat(chunk[:, None], length, axis=1)]
            )
            out, _ = norm_layer(torch.from_numpy(rows).to(model_device))
            states[start : start + len(chunk)] = (
                out[1 : 1 + len(chunk), 0].to(torch.float32).cpu().numpy()
            )
        return np.round(states, 4)

    def find_tristate_matched_params(
        params: dict[str, float], lengths: tuple[int, ...]
    ) -> dict[str, float]:
        """Different raw param values whose constant rows normalize identically."""
        from fusiontimeseries.benchmarking.few_shot.operating_params import normalize_params
        from fusiontimeseries.lib.config import FTSConfig

        op_ranges = FTSConfig().op_ranges
        normalized = normalize_params(params)
        matched: dict[str, float] = {}
        grid = np.linspace(0.02, 0.98, 1921)
        for name in OP_NAMES:
            v = float(normalized[name])
            lo, hi = op_ranges[name]
            raws = lo + grid * (hi - lo)
            roundtrips = (raws - lo) / (hi - lo)  # what normalize_params yields
            # Element 0 = the true value; the rest = candidates (exact float32
            # values as build_op_channels will produce them).
            candidates = np.concatenate([[v], roundtrips]).astype(np.float32)
            states_matrix = np.stack(
                [tri_states_on_device(candidates, length) for length in lengths], axis=1
            )  # (n_candidates+1, n_lengths)
            ok = np.all(states_matrix[1:] == states_matrix[0], axis=1) & (
                np.abs(roundtrips - v) >= 0.15  # demand a genuinely different value
            )
            indices = np.flatnonzero(ok)
            assert len(indices), f"S1: no tri-state-matched value found for {name}"
            best = indices[np.argmax(np.abs(roundtrips[indices] - v))]
            matched[name] = float(raws[best])
        return matched

    # Channel-row finite lengths seen across the k=0 concat rollout (80/144/208)
    # and the k=2 group rollout (267/272); NaN padding does not change nanmean.
    lengths = (80, 144, 208, 267, 272)
    matched_params = find_tristate_matched_params(true_params, lengths)
    print(f"  tri-state-matched params (≠ values, ≡ post-norm rows): {matched_params}")

    # S1a — value erasure + reproducibility (zeroshot+cov)
    fn_a = make_chronos2_covariate_forecast_fn(
        pipeline, "shared", "step", query_params_override=true_params
    )
    out_a = fn_a(trace, [], make_config(0))
    out_a2 = fn_a(trace, [], make_config(0))
    rel_repro = rel_l2(out_a, out_a2)
    assert rel_repro < 1e-5, f"S1a: not reproducible in-process (rel L2 {rel_repro:.2e})"
    out_c = make_chronos2_covariate_forecast_fn(
        pipeline, "shared", "step", query_params_override=matched_params
    )(trace, [], make_config(0))
    rel_ac = rel_l2(out_a, out_c)
    assert rel_ac < 1e-4, (
        f"S1a: tri-state-matched params must give identical outputs "
        f"(rel L2 {rel_ac:.2e}) — the VALUE must be erased"
    )
    out_b = make_chronos2_covariate_forecast_fn(
        pipeline, "shared", "step", query_params_override=opposite
    )(trace, [], make_config(0))
    rel_ab = rel_l2(out_a, out_b)
    assert rel_ab < 0.5, f"S1a: artifact magnitude implausibly large ({rel_ab:.3f})"
    print(
        f"✓ S1a: value erased — matched params rel L2 {rel_ac:.2e} (repro "
        f"{rel_repro:.2e}); tri-state artifact A vs B rel L2 {rel_ab:.4f} "
        f"(±1-bit flips, NOT parameter information)"
    )

    # S1b — row-presence offset: zeroshot+cov vs zeroshot (measured, the
    # grid's zeroshot__shared-opcov anchor quantifies it over all 11 traces)
    out_plain = fns["shared"](trace, [], make_config(0))
    rel_rows = rel_l2(out_a, out_plain)
    assert np.all(np.isfinite(out_a)) and rel_rows < 0.5, (
        f"S1b: row-presence offset {rel_rows:.4f} not finite/sane"
    )
    print(f"✓ S1b: row-presence offset (zeroshot+cov vs zeroshot): rel L2 {rel_rows:.4f}")

    # S1c — group: value erasure for the per-task constant channels
    examples = runner.select_fns["ctx_euclid"](runner.pool, 2, 42, trace[:80], "iteration_8_ifft")
    g_a = make_chronos2_group_covariate_forecast_fn(
        pipeline, query_params_override=true_params
    )(trace, examples, make_config(2))
    g_c = make_chronos2_group_covariate_forecast_fn(
        pipeline, query_params_override=matched_params
    )(trace, examples, make_config(2))
    g_b = make_chronos2_group_covariate_forecast_fn(
        pipeline, query_params_override=opposite
    )(trace, examples, make_config(2))
    rel_gc = rel_l2(g_a, g_c)
    rel_gb = rel_l2(g_a, g_b)
    assert rel_gc < 1e-4, f"S1c: group value not erased (matched rel L2 {rel_gc:.2e})"
    assert rel_gb < 0.5, f"S1c: group artifact implausibly large ({rel_gb:.3f})"
    print(
        f"✓ S1c: group+cov value erased — matched rel L2 {rel_gc:.2e}; "
        f"artifact A vs B rel L2 {rel_gb:.4f}"
    )

    # S2 — path equivalence: dict-task-no-cov vs plain tensor at k=2
    # (licenses running the frozen make_concat_forecast_fn for no-cov cells)
    none_fn = make_chronos2_covariate_forecast_fn(pipeline, "shared", None)
    out_dict = none_fn(trace, examples, make_config(2))
    out_tensor = fns["shared"](trace, examples, make_config(2))
    rel_path = rel_l2(out_dict, out_tensor)
    assert rel_path < 1e-4, f"S2: dict vs tensor path rel L2 {rel_path:.2e} >= 1e-4"
    print(f"✓ S2: dict-no-cov ≡ plain-tensor path (rel L2 {rel_path:.2e})")

    # S3 — sensitivity: with k=5 step channels, query params must be visible
    examples5 = runner.select_fns["ctx_euclid"](runner.pool, 5, 42, trace[:80], "iteration_8_ifft")
    out_true = make_chronos2_covariate_forecast_fn(
        pipeline, "shared", "step", query_params_override=true_params
    )(trace, examples5, make_config(5))
    out_opp = make_chronos2_covariate_forecast_fn(
        pipeline, "shared", "step", query_params_override=opposite
    )(trace, examples5, make_config(5))
    rel_sens = rel_l2(out_true, out_opp)
    assert rel_sens > 1e-6, f"S3: channels invisible (rel L2 {rel_sens:.2e})"
    print(f"✓ S3: channels visible at k=5 — true vs opposite query params rel L2 {rel_sens:.4f}")

    # S4 — NaN mid-stream: force-include the params-less pool example
    paramless = [ex for ex in runner.pool if ex.operating_params is None]
    assert len(paramless) == 1, f"Expected 1 params-less example, got {len(paramless)}"
    mixed: list[FewShotExample] = [paramless[0], examples5[-1]]
    out_nan = fns["cov"](trace, mixed, make_config(2))
    assert np.all(np.isfinite(out_nan)), "S4: non-finite forecast with NaN channel segment"
    print(
        f"✓ S4: params-less example (raw {paramless[0].trace_id}) mid-stream: finite "
        f"(tail mean {np.mean(out_nan[-80:]):.2f})"
    )

    # S5 — timing: one k=10 +cov trace-forecast, extrapolate the grid cost
    examples10 = runner.select_fns["ctx_euclid"](
        runner.pool, 10, 42, trace[:80], "iteration_8_ifft"
    )
    t0 = time.perf_counter()
    out10 = fns["cov"](trace, examples10, make_config(10))
    dt = time.perf_counter() - t0
    assert np.all(np.isfinite(out10)), "S5: non-finite k=10 +cov forecast"
    det_cell = 11 * dt
    random_cell = 11 * args.random_seeds * dt
    print(
        f"✓ S5: k=10 +cov single trace {dt:.1f}s -> deterministic cell ≈ "
        f"{det_cell / 60:.1f} min, {args.random_seeds}-seed random cell ≈ "
        f"{random_cell / 60:.0f} min"
    )
    print("\n✅ Smoke test passed (no results written)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-4 covariate-conditioning grid")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="Which stages to run",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=None,
        help="Override the per-stage default ks (applies to main/perm/group)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=MAIN_STRATEGIES,
        default=None,
        help="Restrict the main stage's strategies (pilot runs)",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=20,
        help="Number of seeds (0..n-1) for the random strategy",
    )
    parser.add_argument(
        "--group-random-seeds",
        type=int,
        default=5,
        help="Number of seeds for the group block's random cells",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the S1-S5 sanity checks and exit (writes nothing)",
    )
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()

    runner = CellRunner(args)
    print(f"Loading {CHRONOS2_SLUG} on {args.device}", flush=True)
    pipeline = make_chronos2_pipeline(args.device)
    fns = {
        # Frozen Phase-3 no-cov path (licensed by smoke S2)
        "shared": make_concat_forecast_fn(chronos2_predict_from_pipeline(pipeline), "shared"),
        "cov": make_chronos2_covariate_forecast_fn(pipeline, "shared", "step"),
        "perm": make_chronos2_covariate_forecast_fn(pipeline, "shared", "permuted"),
        "group": make_chronos2_group_forecast_fn(pipeline),
        "group_cov": make_chronos2_group_covariate_forecast_fn(pipeline, "step"),
    }

    if args.smoke:
        smoke(runner, pipeline, fns, args)
        return

    stages = {
        "anchors": stage_anchors,
        "main": stage_main,
        "k1": stage_k1,
        "perm": stage_perm,
        "group": stage_group,
    }
    t0 = time.perf_counter()
    for stage in args.stages:
        stages[stage](runner, fns, args)
    runner.cleanup(pipeline, fns)
    print(
        f"\n✅ Covariates grid stage(s) complete [{(time.perf_counter() - t0) / 60:.0f} min]",
        flush=True,
    )


if __name__ == "__main__":
    main()
