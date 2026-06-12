"""Phase-7 mechanism analysis: where do the few-shot/finetuning gains come from?

Consumes the full-forecast dumps in ``results/few_shot_v7_mechanism/``
(``run_mechanism_dump.py``) and the Phase-6 scalar results in
``results/few_shot_v6_finetuned/``, and produces, in ``docs/results/fewshot/``:

- ``mechanism_table.md`` — four analysis blocks + verdict:
  1. ERROR DECOMPOSITION: per (seed, trace) tail-MSE split into squared level
     bias ``b² = (mean(ŷ)−mean(y))²`` and squared fluctuation error
     ``e_fluc² = mean(((ŷ−m̂)−(y−ȳ))²)`` — an EXACT identity
     ``mean((ŷ−y)²) = b² + e_fluc²`` over the 80-step tail (cross term
     vanishes; self-tested). Level share, std-ratio ``r_σ = std(ŷ)/std(y)``,
     chaos-floor annotation ``σ_y·√2`` (two independent same-statistics
     realizations), and a claim test per config vs its k=0 anchor.
  2. ROLLOUT STABILITY: tracking horizon ``H_track = min{t : ε̃(t) > τ}`` on
     the normalized error ``ε(t) = |ŷ−y|/σ_y`` (centered moving mean L=16;
     τ=1.0 headline with 0.5/1.5 sensitivity; right-censored at 186;
     64-step-chunk quantization stated) + invariant statistics: tail ACF
     distance (lags 1–20), correlation time τ_c (first lag < 1/e), and a
     flatline detector ``r_σ < 0.2``. Sliding-window Pearson was REJECTED:
     saturated turbulence gives r≈0 even for perfect-statistics forecasts.
  3. ORACLE-GAP: per (query, strategy, pick) level distance
     ``d_lvl = |tail(ex) − true_tail(q)|``, context distance, param distance
     (v6 example_ids + pool; NO model runs); oracle-pick ranks under ctx/op
     rankings; pool-wide Spearman ρ(d_ctx, d_lvl) / ρ(d_op, d_lvl) per query;
     a context-side feature hunt (can any cheap context feature see the tail
     level?) with an OFFLINE feature-knn simulation and a clearly-labeled
     extrapolated gap closure.
  4. PER-TRACE BREAKDOWN + forecast grids for 7 headline configs.
- ``mechanism_decomposition.png``, ``mechanism_horizon.png``,
  ``mechanism_acf.png``, ``mechanism_oracle_gap.png``,
  ``forecast_grid_<config>_{id,ood}.png``.

At load, every dump cell's stored forecasts are re-asserted against its own
recorded scalars (tail means recompute bit-exactly under ``forecast_dtype``).

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_mechanism
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_mechanism --self-test
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
    FewShotExample,
    create_example_pool,
)
from fusiontimeseries.benchmarking.few_shot.finetuned import (
    FINETUNED_SLUG,
    FT_TRAIN_CONTEXT,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    ID_TEST_RAW_IDS,
    get_params_for_benchmark_trace,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import MODEL_SLUGS
from fusiontimeseries.benchmarking.few_shot.run_mechanism_dump import (
    build_reference_index,
    method_label,
)
from fusiontimeseries.benchmarking.few_shot.selection import (
    _params_vector,
    _rank_context_nn,
    _rank_op_knn,
    _zscore,
    estimate_growth_rate,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    IN_DISTRIBUTION_ITERATIONS,
    OUT_OF_DISTRIBUTION_ITERATIONS,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_DUMP_DIR: Path = REPO_ROOT / "results" / "few_shot_v7_mechanism"
DEFAULT_V6_DIR: Path = REPO_ROOT / "results" / "few_shot_v6_finetuned"
DEFAULT_OUT_DIR: Path = REPO_ROOT / "docs" / "results" / "fewshot"

BASE_SLUG: str = MODEL_SLUGS["chronos2"]
BOLT_SLUG: str = MODEL_SLUGS["chronos_bolt"]

W: int = 80  # evaluation tail window (= relevant_prediction_tail)
CTX: int = 80  # ground-truth context steps copied into every forecast
SMOOTH_L: int = 16  # centered moving-mean window for the tracking error
TAUS: tuple[float, ...] = (0.5, 1.0, 1.5)  # horizon thresholds (1.0 headline)
HORIZON_MAX: int = 186  # right-censoring limit (266 - 80 forecast steps)
ACF_SUMMARY_LAGS: int = 20  # d_acf summarized over lags 1..20 (noisy above)
ACF_PLOT_LAGS: int = 40  # figure shows lags 1..40
FLATLINE_RSIGMA: float = 0.2  # r_sigma below this counts as a flatline

SPLITS: tuple[str, ...] = ("in_distribution", "out_of_distribution")
SPLIT_SHORT: dict[str, str] = {"in_distribution": "ID", "out_of_distribution": "OOD"}

#: Dump cells: (label, method, k, anchor label for the claim test).
DUMP_CONFIGS: tuple[tuple[str, str, int, str | None], ...] = (
    ("base k0", method_label(BASE_SLUG, "zeroshot"), 0, None),
    ("base mmr5", method_label(BASE_SLUG, "mmr_euclid"), 5, "base k0"),
    ("base oracle10", method_label(BASE_SLUG, "oracle_tail"), 10, "base k0"),
    ("bolt k0", method_label(BOLT_SLUG, "zeroshot"), 0, None),
    ("bolt mmr10", method_label(BOLT_SLUG, "mmr_euclid"), 10, "bolt k0"),
    ("ft k0", method_label(FINETUNED_SLUG, "zeroshot"), 0, None),
    ("ft mmr5", method_label(FINETUNED_SLUG, "mmr_euclid"), 5, "ft k0"),
    ("ft oracle10", method_label(FINETUNED_SLUG, "oracle_tail"), 10, "ft k0"),
    ("ft random10", method_label(FINETUNED_SLUG, "random"), 10, "ft k0"),
    (
        "ft mmr5@512",
        method_label(FINETUNED_SLUG, "mmr_euclid", FT_TRAIN_CONTEXT),
        5,
        "ft k0",
    ),
    (
        "ft oracle10@512",
        method_label(FINETUNED_SLUG, "oracle_tail", FT_TRAIN_CONTEXT),
        10,
        "ft k0",
    ),
    ("persistence", "persistence", 0, None),
    ("knn_copy5", "knn_copy_k5", 0, None),
)

#: Forecast-grid configs (label -> filename slug).
GRID_CONFIGS: dict[str, str] = {
    "base k0": "base_k0",
    "ft k0": "ft_k0",
    "ft mmr5": "ft_mmr5",
    "ft mmr5@512": "ft_mmr5_win512",
    "ft oracle10": "ft_oracle10",
    "ft oracle10@512": "ft_oracle10_win512",
    "persistence": "persistence",
}

#: Curve subsets for the horizon/ACF figures (13 lines would be unreadable).
CURVE_CONFIGS: tuple[str, ...] = (
    "base k0",
    "ft k0",
    "ft mmr5@512",
    "ft oracle10",
    "persistence",
    "knn_copy5",
)

#: Oracle-gap strategies (ft model, mean decoding, full window).
GAP_CONFIGS: tuple[tuple[str, int], ...] = (
    ("ctx_euclid", 5),
    ("ctx_euclid", 10),
    ("mmr_euclid", 5),
    ("mmr_euclid", 10),
    ("op_knn", 5),
    ("op_knn", 10),
    ("random", 10),
    ("oracle_tail", 10),
)

GROUP_COLORS: dict[str, str] = {
    "base": "tab:blue",
    "bolt": "tab:green",
    "ft": "tab:orange",
    "baseline": "0.45",
}


def group_of(label: str) -> str:
    return "baseline" if label in ("persistence", "knn_copy5") else label.split()[0]


def split_of(trace_key: str) -> str:
    return "out_of_distribution" if trace_key.startswith("ood_") else "in_distribution"


def short_key(trace_key: str) -> str:
    return (
        trace_key.replace("iteration_", "it_")
        .replace("_ifft", "")
        .replace("_realpotens", "")
    )


########################################################
# Core statistics (self-tested)
########################################################


def decompose(y: NDArray, f: NDArray, w: int = W) -> dict[str, float]:
    """Exact tail-MSE decomposition into level bias and fluctuation error.

    Over the last ``w`` steps, with ``ȳ = mean(y)`` and ``m̂ = mean(ŷ)``:
    ``b = m̂ − ȳ`` (the level bias — identical to the result JSONs' ``error``)
    and ``e_fluc = RMSE((ŷ−m̂) − (y−ȳ))``. The cross term vanishes because
    both fluctuation series are mean-zero over the window, so
    ``mean((ŷ−y)²) = b² + e_fluc²`` exactly.

    Args:
        y: Ground-truth series (full trace; only the tail is used).
        f: Forecast series aligned with ``y``.
        w: Tail window length.

    Returns:
        Dict with ``bias``, ``e_fluc``, ``mse``, ``sigma_y`` (truth tail std)
        and ``r_sigma`` (forecast/truth tail std ratio; 0 for a flatline).
    """
    yt = np.asarray(y, dtype=np.float64)[-w:]
    ft = np.asarray(f, dtype=np.float64)[-w:]
    bias = float(ft.mean() - yt.mean())
    fluc = (ft - ft.mean()) - (yt - yt.mean())
    e_fluc = float(np.sqrt(np.mean(fluc**2)))
    mse = float(np.mean((ft - yt) ** 2))
    sigma_y = float(yt.std())
    r_sigma = float(ft.std() / sigma_y) if sigma_y > 0 else float("nan")
    return {
        "bias": bias,
        "e_fluc": e_fluc,
        "mse": mse,
        "sigma_y": sigma_y,
        "r_sigma": r_sigma,
    }


def smoothed_tracking_error(
    y: NDArray, f: NDArray, start: int = CTX, smooth: int = SMOOTH_L
) -> NDArray[np.float64]:
    """Centered moving mean of ``ε(t) = |ŷ(t)−y(t)| / σ_y`` over the forecast.

    ``σ_y`` is the truth's tail std (the natural fluctuation scale); the
    moving mean is edge-corrected (divides by the actual window mass).

    Args:
        y: Ground-truth series.
        f: Forecast aligned with ``y`` (first ``start`` steps are the copied
            context and are excluded).
        start: Forecast start index.
        smooth: Moving-mean window length.

    Returns:
        Smoothed ε̃ over the forecast region (length ``len(y) − start``).
    """
    y64 = np.asarray(y, dtype=np.float64)
    f64 = np.asarray(f, dtype=np.float64)
    sigma_y = float(y64[-W:].std())
    eps = np.abs(f64[start:] - y64[start:]) / max(sigma_y, 1e-12)
    kernel = np.ones(smooth)
    return np.convolve(eps, kernel, mode="same") / np.convolve(
        np.ones_like(eps), kernel, mode="same"
    )


def tracking_horizon(eps_smooth: NDArray, tau: float) -> int | None:
    """First forecast step where ε̃ exceeds τ; None = right-censored."""
    exceed = np.nonzero(eps_smooth > tau)[0]
    return int(exceed[0]) if len(exceed) else None


def acf(x: NDArray, max_lag: int) -> NDArray[np.float64]:
    """Biased sample ACF (lags 1..max_lag) of a mean-removed series.

    The biased estimator (denominator n, full-window variance) is standard
    for short windows. A zero-variance (flatline) series gets an all-zero
    ACF — it has no autocorrelation structure; the ``r_σ`` flatline detector
    flags these separately.

    Args:
        x: Input series (n = 80 tail in this module).
        max_lag: Largest lag.

    Returns:
        Array of shape ``(max_lag,)`` with ρ̂(1)..ρ̂(max_lag).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        return np.zeros(max_lag)
    return np.array(
        [float(np.sum(x[:-lag] * x[lag:])) / denom for lag in range(1, max_lag + 1)]
    )


def correlation_time(rho: NDArray, threshold: float = 1.0 / np.e) -> int | None:
    """First lag where the ACF drops below 1/e; None if it never does."""
    below = np.nonzero(np.asarray(rho) < threshold)[0]
    return int(below[0]) + 1 if len(below) else None


########################################################
# Dump loading (re-asserts consistency at load)
########################################################


def load_dump_cells(dump_dir: Path) -> dict[str, dict]:
    """Load all dump cells; re-assert stored forecasts against the scalars.

    Returns:
        label -> dump dict, augmented with ``_forecasts`` ((seed, trace_key)
        -> float64 array), ``_truths`` (trace_key -> float64 array),
        ``_label`` and ``_anchor``.
    """
    available: dict[tuple[str, int], dict] = {}
    for path in sorted(dump_dir.glob("*_mechanism_dump.json")):
        data = json.load(open(path))
        data["_path"] = str(path)
        key = (data["method"], int(data["config"]["k_shot"]))
        if key not in available or data["timestamp"] > available[key]["timestamp"]:
            available[key] = data

    cells: dict[str, dict] = {}
    for label, method, k, anchor in DUMP_CONFIGS:
        data = available.get((method, k))
        assert data is not None, f"Missing dump cell {label!r} ({method} k={k})"
        dtype = np.dtype(data["forecast_dtype"])
        truths = {
            key: np.asarray(values, dtype=np.float64)
            for key, values in data["ground_truths"].items()
        }
        forecasts: dict[tuple[int, str], NDArray[np.float64]] = {}
        for seed_data in data["per_seed"]:
            for tr in seed_data["per_trace"]:
                typed = np.asarray(
                    seed_data["forecasts"][tr["trace_key"]], dtype=dtype
                )
                tail = float(np.mean(typed[-W:]))
                assert tail == tr["pred_tail_mean"], (
                    f"{label}/{tr['trace_key']}: stored forecast tail does not "
                    f"recompute bit-exactly ({tail!r} vs {tr['pred_tail_mean']!r})"
                )
                assert tr["error"] == tr["pred_tail_mean"] - tr["true_tail_mean"]
                forecasts[(seed_data["seed"], tr["trace_key"])] = typed.astype(
                    np.float64
                )
        assert len(truths) == 11
        data["_forecasts"] = forecasts
        data["_truths"] = truths
        data["_label"] = label
        data["_anchor"] = anchor
        cells[label] = data
    return cells


def iter_seed_traces(cell: dict):
    """Yield (seed, trace_key, truth, forecast, per_trace_record)."""
    for seed_data in cell["per_seed"]:
        for tr in seed_data["per_trace"]:
            key = tr["trace_key"]
            yield (
                seed_data["seed"],
                key,
                cell["_truths"][key],
                cell["_forecasts"][(seed_data["seed"], key)],
                tr,
            )


########################################################
# Block 1 — error decomposition
########################################################


def collect_decomposition(cells: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Aggregate the decomposition per (config, split) + per-point records."""
    rows: list[dict] = []
    points: list[dict] = []
    for label, cell in cells.items():
        per_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
        for seed, key, truth, forecast, tr in iter_seed_traces(cell):
            d = decompose(truth, forecast)
            # The recorded error used the float32 truth-tail mean; decompose
            # works in float64 — agreement to 1e-3 absolute (values O(10-100)).
            assert abs(d["bias"] - tr["error"]) < 1e-3, (
                f"{label}/{key}: decomposition bias {d['bias']:.6f} != "
                f"recorded error {tr['error']:.6f}"
            )
            per_split[split_of(key)].append(d)
            points.append(
                {
                    "label": label,
                    "split": split_of(key),
                    "b2": d["bias"] ** 2,
                    "e2": d["e_fluc"] ** 2,
                }
            )
        for split, records in per_split.items():
            mean_b2 = float(np.mean([d["bias"] ** 2 for d in records]))
            mean_e2 = float(np.mean([d["e_fluc"] ** 2 for d in records]))
            mean_floor2 = float(np.mean([2.0 * d["sigma_y"] ** 2 for d in records]))
            rows.append(
                {
                    "label": label,
                    "split": split,
                    "n": len(records),
                    "mean_b2": mean_b2,
                    "mean_e2": mean_e2,
                    "rmse_pointwise": float(np.sqrt(mean_b2 + mean_e2)),
                    "level_share": mean_b2 / (mean_b2 + mean_e2),
                    "mean_rsigma": float(np.mean([d["r_sigma"] for d in records])),
                    "floor_ratio": float(np.sqrt(mean_e2 / mean_floor2)),
                }
            )
    return rows, points


def decomposition_claim_rows(rows: list[dict]) -> list[dict]:
    """Per config vs its k=0 anchor: which term absorbs the improvement?"""
    indexed = {(r["label"], r["split"]): r for r in rows}
    claim: list[dict] = []
    for label, _method, _k, anchor in DUMP_CONFIGS:
        if anchor is None:
            continue
        for split in SPLITS:
            r, a = indexed[(label, split)], indexed[(anchor, split)]
            d_mse = (a["mean_b2"] + a["mean_e2"]) - (r["mean_b2"] + r["mean_e2"])
            d_b2 = a["mean_b2"] - r["mean_b2"]
            d_e2 = a["mean_e2"] - r["mean_e2"]
            claim.append(
                {
                    "label": label,
                    "anchor": anchor,
                    "split": split,
                    "d_mse": d_mse,
                    "d_b2": d_b2,
                    "d_e2": d_e2,
                    "level_absorbed": d_b2 / d_mse if abs(d_mse) > 1e-9 else float("nan"),
                }
            )
    return claim


def make_decomposition_block(rows: list[dict], claim: list[dict]) -> list[str]:
    lines = [
        "\n## 1. Error decomposition — level bias vs fluctuation error",
        "",
        f"Per (seed, trace), the pointwise tail MSE over the last {W} steps "
        "splits EXACTLY into squared level bias and squared fluctuation "
        "error: `mean((ŷ−y)²) = b² + e_fluc²` with `b = mean(ŷ)−mean(y)` "
        "(identical to the result JSONs' `error`, asserted at load) and "
        "`e_fluc = RMSE((ŷ−m̂)−(y−ȳ))`. NOTE: `rmse_pt` below is this "
        "pointwise tail RMSE, NOT the benchmark's tail-MEAN RMSE — the "
        "benchmark metric sees only `b`. The chaos floor `σ_y·√2` is the "
        "expected `e_fluc` of an independent realization with the same "
        "statistics (two uncorrelated mean-zero series); `e_fluc/floor ≈ 1` "
        "means the forecast fluctuates like an independent sample (no phase "
        "tracking), `1/√2 ≈ 0.71` is an exact flatline, `> 1` means excess "
        "fluctuation. `r_σ = std(ŷ)/std(y)` checks invariant amplitude.",
        "",
        "| config | split | n | rmse_pt | mean b² | mean e_fluc² | level share "
        "| r_σ | e_fluc/floor |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {SPLIT_SHORT[r['split']]} | {r['n']} | "
            f"{r['rmse_pointwise']:.2f} | {r['mean_b2']:.1f} | {r['mean_e2']:.1f} | "
            f"{r['level_share']:.2f} | {r['mean_rsigma']:.2f} | "
            f"{r['floor_ratio']:.2f} |"
        )
    lines += [
        "",
        "Claim test — per config vs its k=0 anchor, how much of the tail-MSE "
        "improvement is absorbed by the LEVEL term (Δb²/ΔMSE; ≈1 ⇒ the gain "
        "is pure level calibration):",
        "",
        "| config | anchor | split | ΔMSE (anchor−config) | Δb² | Δe_fluc² | "
        "level absorbed |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in claim:
        lines.append(
            f"| {c['label']} | {c['anchor']} | {SPLIT_SHORT[c['split']]} | "
            f"{c['d_mse']:+.1f} | {c['d_b2']:+.1f} | {c['d_e2']:+.1f} | "
            f"{c['level_absorbed']:.2f} |"
        )
    return lines


def fig_decomposition(
    rows: list[dict], points: list[dict], out_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    labels = [label for label, *_ in DUMP_CONFIGS]
    for ax, split in zip(axes[:2], SPLITS):
        split_rows = {r["label"]: r for r in rows if r["split"] == split}
        x = np.arange(len(labels))
        b2 = np.array([split_rows[label]["mean_b2"] for label in labels])
        e2 = np.array([split_rows[label]["mean_e2"] for label in labels])
        ax.bar(x, b2, color="tab:red", label="level bias² (b²)")
        ax.bar(x, e2, bottom=b2, color="tab:gray", label="fluctuation² (e_fluc²)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{SPLIT_SHORT[split]}: mean tail MSE = b² + e_fluc²")
        ax.set_ylabel("mean squared error")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8, frameon=False)
    ax = axes[2]
    for grp, color in GROUP_COLORS.items():
        pts = [p for p in points if group_of(p["label"]) == grp]
        ax.scatter(
            [max(p["e2"], 1e-3) for p in pts],
            [max(p["b2"], 1e-3) for p in pts],
            s=12,
            alpha=0.55,
            color=color,
            label=grp,
        )
    lims = (1e-3, 1e5)
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("e_fluc² (per seed-trace)")
    ax.set_ylabel("b² (per seed-trace)")
    ax.set_title("per-trace: bias² vs fluctuation²")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.suptitle(
        "Tail-MSE decomposition: the benchmark metric sees only the red term",
        y=1.0,
    )
    fig.tight_layout()
    path = out_dir / "mechanism_decomposition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


########################################################
# Block 2 — horizon + invariant statistics
########################################################


def collect_horizons(
    cells: dict[str, dict],
) -> tuple[list[dict], dict[str, dict[str, NDArray]]]:
    """Horizon rows per (config, split, tau) + median ε̃ curves per config."""
    rows: list[dict] = []
    curves: dict[str, dict[str, NDArray]] = {}
    for label, cell in cells.items():
        eps_by_split: dict[str, list[NDArray]] = {split: [] for split in SPLITS}
        for _seed, key, truth, forecast, _tr in iter_seed_traces(cell):
            eps_by_split[split_of(key)].append(
                smoothed_tracking_error(truth, forecast)
            )
        curves[label] = {
            split: np.median(np.stack(arrs), axis=0)
            for split, arrs in eps_by_split.items()
        }
        for split, arrs in eps_by_split.items():
            for tau in TAUS:
                horizons = [tracking_horizon(eps, tau) for eps in arrs]
                censored = sum(h is None for h in horizons)
                filled = [HORIZON_MAX if h is None else h for h in horizons]
                rows.append(
                    {
                        "label": label,
                        "split": split,
                        "tau": tau,
                        "median_h": float(np.median(filled)),
                        "n_censored": censored,
                        "n": len(filled),
                    }
                )
    return rows, curves


def collect_acf_stats(
    cells: dict[str, dict],
) -> tuple[list[dict], dict[str, dict[str, NDArray]], dict[str, NDArray], dict]:
    """ACF distance/τ_c/flatline rows + mean ACF curves (+ truth reference)."""
    truth_acfs: dict[str, NDArray] = {}
    any_cell = next(iter(cells.values()))
    truth_tau_c: dict[str, list[int | None]] = {split: [] for split in SPLITS}
    truth_curves: dict[str, NDArray] = {}
    by_split: dict[str, list[NDArray]] = {split: [] for split in SPLITS}
    for key, truth in any_cell["_truths"].items():
        rho = acf(truth[-W:], ACF_PLOT_LAGS)
        truth_acfs[key] = rho
        by_split[split_of(key)].append(rho)
        truth_tau_c[split_of(key)].append(correlation_time(rho))
    for split, arrs in by_split.items():
        truth_curves[split] = np.mean(np.stack(arrs), axis=0)
    truth_info = {
        split: {
            "median_tau_c": float(
                np.median([t for t in truth_tau_c[split] if t is not None])
            ),
            "n_censored": sum(t is None for t in truth_tau_c[split]),
        }
        for split in SPLITS
    }

    rows: list[dict] = []
    curves: dict[str, dict[str, NDArray]] = {}
    for label, cell in cells.items():
        acc: dict[str, dict[str, list]] = {
            split: {"d_acf": [], "tau_c": [], "flatline": [], "rho": []}
            for split in SPLITS
        }
        for _seed, key, truth, forecast, _tr in iter_seed_traces(cell):
            split = split_of(key)
            rho_f = acf(forecast[-W:], ACF_PLOT_LAGS)
            rho_y = truth_acfs[key]
            acc[split]["rho"].append(rho_f)
            acc[split]["d_acf"].append(
                float(
                    np.mean(
                        np.abs(
                            rho_f[:ACF_SUMMARY_LAGS] - rho_y[:ACF_SUMMARY_LAGS]
                        )
                    )
                )
            )
            acc[split]["tau_c"].append(correlation_time(rho_f))
            d = decompose(truth, forecast)
            acc[split]["flatline"].append(d["r_sigma"] < FLATLINE_RSIGMA)
        curves[label] = {
            split: np.mean(np.stack(acc[split]["rho"]), axis=0) for split in SPLITS
        }
        for split in SPLITS:
            tau_cs = [t for t in acc[split]["tau_c"] if t is not None]
            rows.append(
                {
                    "label": label,
                    "split": split,
                    "mean_d_acf": float(np.mean(acc[split]["d_acf"])),
                    "median_tau_c": float(np.median(tau_cs)) if tau_cs else float("nan"),
                    "tau_c_censored": sum(t is None for t in acc[split]["tau_c"]),
                    "n_flatline": int(np.sum(acc[split]["flatline"])),
                    "n": len(acc[split]["d_acf"]),
                }
            )
    return rows, curves, truth_curves, truth_info


def make_horizon_block(
    horizon_rows: list[dict], acf_rows: list[dict], truth_info: dict
) -> list[str]:
    lines = [
        "\n## 2. Rollout stability — tracking horizon and invariant statistics",
        "",
        "Does ICL/finetuning extend the usable autoregressive horizon? "
        f"`ε(t) = |ŷ(t)−y(t)|/σ_y` (σ_y = truth tail std), smoothed with a "
        f"centered L={SMOOTH_L} moving mean; `H_track` = first forecast step "
        f"where ε̃ > τ. Right-censored at {HORIZON_MAX} steps (counts "
        "reported); onsets are quantized by the 64-step rollout chunks. "
        "Sliding-window Pearson was rejected: on saturated turbulence even a "
        "perfect-statistics forecast decorrelates (r≈0), so correlation "
        "measures phase luck, not usefulness. CAVEAT — on a chaotic "
        "saturated series, ε̃ of a perfect-statistics forecast hovers near "
        "√2 ≈ 1.41, so τ=1.0 horizons are short for EVERY method by "
        "construction; the invariant-statistics block below is the primary "
        "stability evidence. The GyroSwin paper's ~110-step correlation time "
        "(vs ~7–10 for baselines) is on 5D states with a different metric — "
        "the comparison is QUALITATIVE only (our traces are additionally "
        "1/3-subsampled).",
        "",
        "| config | split | H@τ=0.5 | H@τ=1.0 | H@τ=1.5 | censored@1.0 | n |",
        "|---|---|---|---|---|---|---|",
    ]
    indexed = {(r["label"], r["split"], r["tau"]): r for r in horizon_rows}
    for label, *_ in DUMP_CONFIGS:
        for split in SPLITS:
            cells_ = [indexed[(label, split, tau)] for tau in TAUS]
            headline = indexed[(label, split, 1.0)]
            lines.append(
                f"| {label} | {SPLIT_SHORT[split]} | "
                + " | ".join(f"{c['median_h']:.0f}" for c in cells_)
                + f" | {headline['n_censored']}/{headline['n']} | {headline['n']} |"
            )
    lines += [
        "",
        "Invariant statistics on the mean-removed 80-step tail (the "
        "context-parroting literature's claim: TSFMs preserve invariant "
        "statistics after point tracking fails): `d_acf` = mean abs ACF "
        f"distance to the truth over lags 1–{ACF_SUMMARY_LAGS} (biased "
        "estimator; lags beyond are figure-only — too noisy at n=80), "
        "`τ_c` = first lag with ρ̂ < 1/e, flatline = share of forecasts with "
        f"`r_σ < {FLATLINE_RSIGMA}` (the crispest stability statistic: a "
        "flatlined rollout has no fluctuation structure at all).",
        "",
        "| config | split | d_acf (1–20) | median τ_c (truth: ID "
        f"{truth_info['in_distribution']['median_tau_c']:.0f} / OOD "
        f"{truth_info['out_of_distribution']['median_tau_c']:.0f}) | "
        "flatlines |",
        "|---|---|---|---|---|",
    ]
    indexed_acf = {(r["label"], r["split"]): r for r in acf_rows}
    for label, *_ in DUMP_CONFIGS:
        for split in SPLITS:
            r = indexed_acf[(label, split)]
            tau_c = (
                f"{r['median_tau_c']:.0f}"
                if np.isfinite(r["median_tau_c"])
                else "censored"
            )
            if r["tau_c_censored"]:
                tau_c += f" ({r['tau_c_censored']} censored)"
            lines.append(
                f"| {label} | {SPLIT_SHORT[split]} | {r['mean_d_acf']:.3f} | "
                f"{tau_c} | {r['n_flatline']}/{r['n']} |"
            )
    return lines


def fig_horizon(
    horizon_rows: list[dict],
    curves: dict[str, dict[str, NDArray]],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    for label in CURVE_CONFIGS:
        curve = curves[label]["in_distribution"]
        ax.plot(
            np.arange(len(curve)) + CTX,
            curve,
            label=label,
            color=GROUP_COLORS[group_of(label)],
            ls={"base": "-", "bolt": ":", "ft": "-", "baseline": "--"}[
                group_of(label)
            ],
            lw=1.4,
            alpha=0.9 if group_of(label) != "ft" else 1.0,
        )
    # distinguish the three ft lines by style
    for line, label in zip(ax.lines, CURVE_CONFIGS):
        if label == "ft k0":
            line.set_linestyle("-.")
        elif label == "ft oracle10":
            line.set_linestyle(":")
    for tau in TAUS:
        ax.axhline(tau, color="0.7", lw=0.7, zorder=0)
    ax.axhline(np.sqrt(2), color="tab:red", lw=0.9, ls="--", alpha=0.7)
    ax.text(262, np.sqrt(2) + 0.04, "√2 (independent realization)", fontsize=7,
            ha="right", color="tab:red")
    ax.set_xlabel("timestep")
    ax.set_ylabel("median ε̃(t) over ID traces")
    ax.set_title("smoothed tracking error (ID)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [label for label, *_ in DUMP_CONFIGS]
    indexed = {(r["label"], r["split"], r["tau"]): r for r in horizon_rows}
    x = np.arange(len(labels))
    width = 0.38
    for off, split, alpha in ((-width / 2, SPLITS[0], 1.0), (width / 2, SPLITS[1], 0.55)):
        values = [indexed[(label, split, 1.0)]["median_h"] for label in labels]
        bars = ax.bar(
            x + off,
            values,
            width,
            color=[GROUP_COLORS[group_of(label)] for label in labels],
            alpha=alpha,
            label=SPLIT_SHORT[split],
        )
        for bar, label in zip(bars, labels):
            r = indexed[(label, split, 1.0)]
            if r["n_censored"]:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"c{r['n_censored']}",
                    fontsize=6,
                    ha="center",
                )
    ax.axhline(HORIZON_MAX, color="0.6", lw=0.8, ls=":")
    ax.text(0.1, HORIZON_MAX + 3, f"censoring limit {HORIZON_MAX}", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("median H_track @ τ=1.0 (steps)")
    ax.set_title("tracking horizon (cN = censored traces)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "mechanism_horizon.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_acf(
    curves: dict[str, dict[str, NDArray]],
    truth_curves: dict[str, NDArray],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    lags = np.arange(1, ACF_PLOT_LAGS + 1)
    for ax, split in zip(axes, SPLITS):
        ax.plot(
            lags, truth_curves[split], color="black", lw=2.2, label="ground truth"
        )
        for label in CURVE_CONFIGS:
            ax.plot(
                lags,
                curves[label][split],
                lw=1.2,
                label=label,
                color=GROUP_COLORS[group_of(label)],
                ls={"base k0": "-", "ft k0": "-.", "ft mmr5@512": "-",
                    "ft oracle10": ":", "persistence": "--", "knn_copy5": "-"}[
                    label
                ],
                alpha=0.9,
            )
        ax.axvline(ACF_SUMMARY_LAGS, color="0.7", lw=0.8, ls=":")
        ax.axhline(1.0 / np.e, color="0.7", lw=0.8)
        ax.text(ACF_PLOT_LAGS - 0.5, 1.0 / np.e + 0.02, "1/e", fontsize=7, ha="right")
        ax.set_xlabel("lag (subsampled steps)")
        ax.set_title(
            f"{SPLIT_SHORT[split]}: mean tail ACF "
            f"(d_acf summarized over lags 1–{ACF_SUMMARY_LAGS})"
        )
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ρ̂(lag), mean-removed 80-step tail")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = out_dir / "mechanism_acf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


########################################################
# Block 3 — oracle gap (v6 scalars + pool; no model runs)
########################################################


def context_features(ctx: NDArray) -> dict[str, float]:
    """Cheap context features a training-free selector could exploit."""
    ctx64 = np.asarray(ctx, dtype=np.float64)
    return {
        "ctx_mean": float(ctx64.mean()),
        "last16_mean": float(ctx64[-16:].mean()),
        "ctx_max": float(ctx64.max()),
        "growth_rate": estimate_growth_rate(ctx64),
        "ctx_std": float(ctx64.std()),
    }


def analyze_oracle_gap(
    v6_index: dict[tuple[str, int], dict],
    pool: list[FewShotExample],
    truths: dict[str, NDArray],
) -> dict:
    """Oracle-gap characterization from v6 example_ids + the pool.

    Returns a dict with pick records, per-(strategy, seed, query) point
    records (improvement over the ft k=0 anchor), oracle-pick ranks,
    per-query Spearman correlations, the feature hunt and the per-trace
    table inputs.
    """
    pool_tails = np.array(
        [float(np.mean(ex.trace_array[-W:])) for ex in pool]
    )
    pool_by_id = {ex.trace_id: i for i, ex in enumerate(pool)}
    pool_zctx = np.stack([_zscore(ex.context_array) for ex in pool])
    pool_features = {
        name: np.array(
            [context_features(ex.context_array)[name] for ex in pool]
        )
        for name in context_features(pool[0].context_array)
    }
    pool_params = [
        _params_vector(ex.operating_params)
        if ex.operating_params is not None
        else None
        for ex in pool
    ]

    ft_k0 = v6_index[(method_label(FINETUNED_SLUG, "zeroshot"), 0)]
    k0_records = {
        tr["trace_key"]: tr for tr in ft_k0["per_seed"][0]["per_trace"]
    }
    queries = list(IN_DISTRIBUTION_ITERATIONS) + list(OUT_OF_DISTRIBUTION_ITERATIONS)
    q_ctx = {key: truths[key][:CTX] for key in queries}
    q_zctx = {key: _zscore(q_ctx[key]) for key in queries}
    q_true = {key: k0_records[key]["true_tail_mean"] for key in queries}
    q_params = {
        key: _params_vector(get_params_for_benchmark_trace(key)) for key in queries
    }
    q_features = {key: context_features(q_ctx[key]) for key in queries}

    pick_rows: list[dict] = []
    point_rows: list[dict] = []
    for strategy, k in GAP_CONFIGS:
        cell = v6_index[(method_label(FINETUNED_SLUG, strategy), k)]
        for seed_data in cell["per_seed"]:
            errors = {tr["trace_key"]: tr for tr in seed_data["per_trace"]}
            for key in queries:
                picks = seed_data["example_ids"][key]
                d_lvls, d_ctxs, d_ops = [], [], []
                for pick_id in picks:
                    i = pool_by_id[pick_id]
                    d_lvl = float(abs(pool_tails[i] - q_true[key]))
                    d_ctx = float(np.linalg.norm(pool_zctx[i] - q_zctx[key]))
                    d_op = (
                        float(np.linalg.norm(pool_params[i] - q_params[key]))
                        if pool_params[i] is not None
                        else None
                    )
                    d_lvls.append(d_lvl)
                    d_ctxs.append(d_ctx)
                    d_ops.append(d_op)
                    pick_rows.append(
                        {
                            "strategy": strategy,
                            "k": k,
                            "seed": seed_data["seed"],
                            "query": key,
                            "split": split_of(key),
                            "pick_id": pick_id,
                            "d_lvl": d_lvl,
                            "rel_lvl": d_lvl / max(1.0, abs(q_true[key])),
                            "d_ctx": d_ctx,
                            "d_op": d_op,
                        }
                    )
                point_rows.append(
                    {
                        "strategy": strategy,
                        "k": k,
                        "seed": seed_data["seed"],
                        "query": key,
                        "split": split_of(key),
                        "min_d_lvl": float(np.min(d_lvls)),
                        "mean_d_lvl": float(np.mean(d_lvls)),
                        "abs_err": abs(errors[key]["error"]),
                        "abs_err_k0": abs(k0_records[key]["error"]),
                        "improvement": abs(k0_records[key]["error"])
                        - abs(errors[key]["error"]),
                    }
                )

    # Oracle picks: pool-min d_lvl assertion + ranks under ctx/op rankings
    oracle_rank_rows: list[dict] = []
    for key in queries:
        oracle_cell = v6_index[(method_label(FINETUNED_SLUG, "oracle_tail"), 10)]
        picks = oracle_cell["per_seed"][0]["example_ids"][key]
        pick_idx = [pool_by_id[p] for p in picks]
        pool_min = float(np.min(np.abs(pool_tails - q_true[key])))
        oracle_min = float(
            np.min([abs(pool_tails[i] - q_true[key]) for i in pick_idx])
        )
        assert oracle_min == pool_min, (
            f"{key}: oracle pick min d_lvl {oracle_min} != pool min {pool_min}"
        )
        ctx_ranking = _rank_context_nn(pool, q_ctx[key], "euclidean")
        ctx_pos = {pool_i: rank for rank, pool_i in enumerate(ctx_ranking)}
        op_ranking = _rank_op_knn(pool, key)
        op_pos = {pool_i: rank for rank, pool_i in enumerate(op_ranking)}
        ctx_ranks = [ctx_pos[i] for i in pick_idx]
        op_ranks = [op_pos[i] for i in pick_idx if i in op_pos]
        valid = [i for i in range(len(pool)) if pool_params[i] is not None]
        rho_ctx = stats.spearmanr(
            np.linalg.norm(pool_zctx - q_zctx[key], axis=1),
            np.abs(pool_tails - q_true[key]),
        ).statistic
        rho_op = stats.spearmanr(
            [np.linalg.norm(pool_params[i] - q_params[key]) for i in valid],
            [abs(pool_tails[i] - q_true[key]) for i in valid],
        ).statistic
        oracle_rank_rows.append(
            {
                "query": key,
                "split": split_of(key),
                "pool_min_d_lvl": pool_min,
                "median_ctx_rank": float(np.median(ctx_ranks)),
                "median_op_rank": float(np.median(op_ranks)),
                "rho_ctx_lvl": float(rho_ctx),
                "rho_op_lvl": float(rho_op),
            }
        )

    # Feature hunt: which cheap context feature sees the tail level?
    feature_rows: list[dict] = []
    for name, values in pool_features.items():
        rho = stats.spearmanr(values, pool_tails).statistic
        feature_rows.append({"feature": name, "rho_tail": float(rho)})
    best_feature = max(feature_rows, key=lambda r: abs(r["rho_tail"]))["feature"]

    featknn_rows: list[dict] = []
    for key in queries:
        scores = np.abs(pool_features[best_feature] - q_features[key][best_feature])
        order = np.argsort(scores, kind="stable")
        for k in (5, 10):
            d_lvls = [abs(pool_tails[i] - q_true[key]) for i in order[:k]]
            featknn_rows.append(
                {
                    "query": key,
                    "split": split_of(key),
                    "k": k,
                    "min_d_lvl": float(np.min(d_lvls)),
                    "mean_d_lvl": float(np.mean(d_lvls)),
                }
            )

    # Extrapolation: fit improvement ~ min_d_lvl on the ID cloud, predict the
    # feature-knn selector's improvement at its achievable min_d_lvl.
    extrapolation: dict[str, dict] = {}
    for split in SPLITS:
        pts = [p for p in point_rows if p["split"] == split]
        x = np.array([p["min_d_lvl"] for p in pts])
        y = np.array([p["improvement"] for p in pts])
        fit = stats.linregress(x, y)
        rho_cloud = stats.spearmanr(x, y).statistic
        split_queries = [q for q in queries if split_of(q) == split]
        predicted_errs = []
        for key in split_queries:
            fk = next(
                r for r in featknn_rows if r["query"] == key and r["k"] == 5
            )
            predicted_improvement = fit.intercept + fit.slope * fk["min_d_lvl"]
            predicted_errs.append(
                max(0.0, abs(k0_records[key]["error"]) - predicted_improvement)
            )
        extrapolation[split] = {
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r2": float(fit.rvalue**2),
            "rho_cloud": float(rho_cloud),
            "predicted_rmse": float(np.sqrt(np.mean(np.square(predicted_errs)))),
        }

    return {
        "pick_rows": pick_rows,
        "point_rows": point_rows,
        "oracle_rank_rows": oracle_rank_rows,
        "feature_rows": feature_rows,
        "best_feature": best_feature,
        "featknn_rows": featknn_rows,
        "extrapolation": extrapolation,
        "pool_tails": pool_tails,
        "q_true": q_true,
        "k0_records": k0_records,
        "queries": queries,
    }


def make_gap_block(gap: dict) -> list[str]:
    lines = [
        "\n## 3. The oracle–legit gap — the pool has the right examples, "
        "retrieval can't see them",
        "",
        "All from v6 `example_ids` + the 245-trace pool (no model runs); "
        "errors are the ft-model mean-decoding cells, improvement is vs the "
        "ft k=0 anchor. `d_lvl` = |example tail mean − TRUE query tail mean| "
        "— what the oracle minimizes and what the metric ultimately scores.",
        "",
        "Pick quality per strategy (over all queries × picks; seeds for "
        "random):",
        "",
        "| strategy | k | median pick d_lvl | median per-query min d_lvl | "
        "median improvement (ID) |",
        "|---|---|---|---|---|",
    ]
    for strategy, k in GAP_CONFIGS:
        picks = [
            r for r in gap["pick_rows"] if (r["strategy"], r["k"]) == (strategy, k)
        ]
        points = [
            r
            for r in gap["point_rows"]
            if (r["strategy"], r["k"]) == (strategy, k)
            and r["split"] == "in_distribution"
        ]
        lines.append(
            f"| {strategy}{' (cheats)' if strategy == 'oracle_tail' else ''} | {k} | "
            f"{np.median([r['d_lvl'] for r in picks]):.2f} | "
            f"{np.median([r['min_d_lvl'] for r in points]):.2f} | "
            f"{np.median([r['improvement'] for r in points]):+.2f} |"
        )
    lines += [
        "",
        "Note: median per-trace improvement understates the RMSE story — the "
        "headline RMSE gains come from fixing the LARGEST-level traces (see "
        "the per-trace table below), not the median trace.",
        "",
        "Why context distance cannot find the oracle's picks — the oracle's "
        "k=10 picks ranked by context/param distance (median rank of 245; "
        "~122 would be chance), and pool-wide Spearman between each distance "
        "and `d_lvl` per query:",
        "",
        "| query | split | pool-min d_lvl | oracle picks' median ctx-rank | "
        "median op-rank | ρ(d_ctx, d_lvl) | ρ(d_op, d_lvl) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in gap["oracle_rank_rows"]:
        lines.append(
            f"| {short_key(r['query'])} | {SPLIT_SHORT[r['split']]} | "
            f"{r['pool_min_d_lvl']:.2f} | {r['median_ctx_rank']:.0f} | "
            f"{r['median_op_rank']:.0f} | {r['rho_ctx_lvl']:+.2f} | "
            f"{r['rho_op_lvl']:+.2f} |"
        )
    med = lambda field: float(  # noqa: E731
        np.median([r[field] for r in gap["oracle_rank_rows"]])
    )
    lines.append(
        f"| **median** | | {med('pool_min_d_lvl'):.2f} | "
        f"{med('median_ctx_rank'):.0f} | {med('median_op_rank'):.0f} | "
        f"{med('rho_ctx_lvl'):+.2f} | {med('rho_op_lvl'):+.2f} |"
    )
    lines += [
        "",
        "Context-side feature hunt — pool-wide Spearman between cheap "
        "context features and the example's OWN tail level (a feature that "
        "sees the level from 80 linear-phase steps would make a training-"
        "free level-matching selector possible):",
        "",
        "| feature | ρ(feature, tail mean) over the pool |",
        "|---|---|",
    ]
    for r in gap["feature_rows"]:
        marker = " ← best" if r["feature"] == gap["best_feature"] else ""
        lines.append(f"| {r['feature']} | {r['rho_tail']:+.3f}{marker} |")
    fk5 = [r for r in gap["featknn_rows"] if r["k"] == 5]
    fk5_id = [r for r in fk5 if r["split"] == "in_distribution"]
    ex = gap["extrapolation"]["in_distribution"]
    lines += [
        "",
        f"OFFLINE feature-knn simulation (selection only, k=5, best feature "
        f"= `{gap['best_feature']}`): median per-query min d_lvl "
        f"{np.median([r['min_d_lvl'] for r in fk5_id]):.2f} ID "
        f"({np.median([r['min_d_lvl'] for r in fk5]):.2f} all). "
        f"EXTRAPOLATION (labeled as such — no model run): the ID cloud's "
        f"linear fit improvement ≈ {ex['intercept']:.1f} + {ex['slope']:.2f}"
        f"·min_d_lvl (R²={ex['r2']:.2f}, Spearman ρ={ex['rho_cloud']:+.2f}) "
        f"predicts a feature-knn ID RMSE of ≈ {ex['predicted_rmse']:.1f}. "
        f"Read this as 'the cloud cannot promise a gain', not as a "
        f"prediction: at R²={ex['r2']:.2f} best-pick level match is NOT a "
        "sufficient statistic of the gain — context COMPOSITION (the "
        "wrong-level example mass the Phase-6 win512 clamp removes) carries "
        "the rest of the variance, so a level-aware selector would have to "
        "be paired with composition control (e.g. drop unmatched picks "
        "rather than fill k) before it could pay off.",
    ]
    return lines


def fig_oracle_gap(gap: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax = axes[0]
    labels, data = [], []
    for strategy, k in GAP_CONFIGS:
        picks = [
            r for r in gap["pick_rows"] if (r["strategy"], r["k"]) == (strategy, k)
        ]
        labels.append(f"{strategy}\nk={k}")
        data.append([max(r["d_lvl"], 1e-2) for r in picks])
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    for i, values in enumerate(data):
        jitter = (np.random.default_rng(i).random(len(values)) - 0.5) * 0.25
        ax.plot(
            np.full(len(values), i + 1) + jitter,
            values,
            ".",
            ms=2.5,
            alpha=0.25,
            color="tab:blue",
        )
    ax.set_yscale("log")
    ax.set_ylabel("pick d_lvl = |example tail − true query tail|")
    ax.set_title("pick quality per strategy (all queries × picks)")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    markers = {"in_distribution": "o", "out_of_distribution": "^"}
    colors = {
        "ctx_euclid": "tab:blue",
        "mmr_euclid": "tab:cyan",
        "op_knn": "tab:purple",
        "random": "0.6",
        "oracle_tail": "tab:red",
    }
    for strategy, k in GAP_CONFIGS:
        for split in SPLITS:
            pts = [
                p
                for p in gap["point_rows"]
                if (p["strategy"], p["k"]) == (strategy, k) and p["split"] == split
            ]
            ax.scatter(
                [max(p["min_d_lvl"], 1e-2) for p in pts],
                [p["improvement"] for p in pts],
                s=18,
                alpha=0.6,
                color=colors[strategy],
                marker=markers[split],
                label=f"{strategy} k={k}" if split == "in_distribution" else None,
            )
    ex = gap["extrapolation"]["in_distribution"]
    xs = np.linspace(0.01, max(p["min_d_lvl"] for p in gap["point_rows"]), 100)
    ax.plot(
        xs,
        ex["intercept"] + ex["slope"] * xs,
        "k--",
        lw=1.0,
        label=f"ID fit (R²={ex['r2']:.2f})",
    )
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("min pick d_lvl (per query, log)")
    ax.set_ylabel("improvement over ft k=0 (|err_k0| − |err|)")
    ax.set_title("improvement vs best-pick level match — weak relation (○ ID, △ OOD)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    fig.suptitle(
        "Oracle gap: retrieval picks are far from the query's true level; "
        "best-pick match only weakly predicts the gain (composition matters)",
        y=1.0,
    )
    fig.tight_layout()
    path = out_dir / "mechanism_oracle_gap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


########################################################
# Block 4 — per-trace table + forecast grids
########################################################


def make_per_trace_block(
    gap: dict, v6_index: dict[tuple[str, int], dict]
) -> list[str]:
    best_legit = v6_index[
        (method_label(FINETUNED_SLUG, "mmr_euclid", FT_TRAIN_CONTEXT), 5)
    ]
    oracle = v6_index[(method_label(FINETUNED_SLUG, "oracle_tail"), 10)]
    legit_err = {
        tr["trace_key"]: tr["error"]
        for tr in best_legit["per_seed"][0]["per_trace"]
    }
    oracle_err = {
        tr["trace_key"]: tr["error"] for tr in oracle["per_seed"][0]["per_trace"]
    }
    pool_min = {r["query"]: r["pool_min_d_lvl"] for r in gap["oracle_rank_rows"]}
    lines = [
        "\n## 4. Per-trace breakdown",
        "",
        "Signed tail-mean errors (pred − true) of the three headline ft "
        "cells, against the best the pool could possibly offer "
        "(`pool-min d_lvl` — by construction the oracle's top pick, "
        "asserted):",
        "",
        "| trace | true tail | ft k0 err | ft mmr5@512 err (best legit) | "
        "ft oracle10 err (cheats) | pool-min d_lvl |",
        "|---|---|---|---|---|---|",
    ]
    for key in gap["queries"]:
        lines.append(
            f"| {short_key(key)} | {gap['q_true'][key]:.2f} | "
            f"{gap['k0_records'][key]['error']:+.2f} | {legit_err[key]:+.2f} | "
            f"{oracle_err[key]:+.2f} | {pool_min[key]:.2f} |"
        )
    return lines


def fig_forecast_grids(cells: dict[str, dict], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for label, slug in GRID_CONFIGS.items():
        cell = cells[label]
        seed = cell["per_seed"][0]["seed"]
        errors = {
            tr["trace_key"]: tr["error"] for tr in cell["per_seed"][0]["per_trace"]
        }
        for split_name, keys, (nrow, ncol) in (
            ("id", list(IN_DISTRIBUTION_ITERATIONS), (3, 2)),
            ("ood", list(OUT_OF_DISTRIBUTION_ITERATIONS), (2, 3)),
        ):
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(4.4 * ncol, 2.9 * nrow), sharex=True
            )
            axes = np.atleast_1d(axes).ravel()
            for ax, key in zip(axes, keys):
                truth = cell["_truths"][key]
                forecast = cell["_forecasts"][(seed, key)]
                ax.plot(truth, color="black", lw=1.0, label="ground truth")
                ax.plot(
                    forecast,
                    color=GROUP_COLORS[group_of(label)],
                    lw=1.1,
                    label=label,
                )
                ax.axvline(CTX, color="0.6", lw=0.8, ls="--")
                ax.set_title(
                    f"{short_key(key)} (Δtail={errors[key]:+.1f})", fontsize=9
                )
                ax.grid(alpha=0.25)
            for ax in axes[len(keys):]:
                ax.axis("off")
            axes[0].legend(fontsize=8, frameon=False)
            fig.suptitle(
                f"{label} — {split_name.upper()} forecasts "
                "(dashed line: forecast start, step 80)",
                y=1.0,
            )
            fig.tight_layout()
            path = out_dir / f"forecast_grid_{slug}_{split_name}.png"
            fig.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


########################################################
# Verdict (data-driven scaffold; prose finalized with the numbers)
########################################################

VERDICT = """
**Verdict — the gains are level calibration; nothing tracks the turbulence,
and the oracle gap is an information limit of the 80-step context, not a
distance-function deficiency.**

(1) **Decomposition: adaptation buys level, nothing else.** In all 16
claim-test rows the level term absorbs 87–112% of the tail-MSE change
(Δb²/ΔMSE ≈ 1 up to per-trace noise) — retrieval-ICL, finetuning, the
window clamp, and even the HARM from random examples are all movements of
the predicted saturation level; the fluctuation error never moves. e_fluc
sits at 0.68–0.89× the chaos floor σ_y·√2 in every one of the 26
config×split rows (an exact flatline gives 0.71): no configuration
phase-tracks the turbulence — exactly the context-parroting prediction.
Since the benchmark metric reads only the level term (√(mean b²) ≡ the
benchmark tail RMSE; ft k0: √492.9 = 22.2), "better RMSE" and "better level
calibration" are the same statement — this is WHY shared scaling (Phase 3)
was the unlock and WHY retrieval works at all. After finetuning, the ID
level share drops to 0.44 (k0) and 0.12 (oracle ICL): on ID the level
problem is largely solved and the residual is irreducible fluctuation. OOD
level share stays ≥ 0.67 for every legit config — OOD remains an unsolved
level problem (only the cheating oracle pushes it to 0.39).

(2) **Rollout stability: no genuine horizon extension; under-dispersion is
universal.** Median tracking horizons at τ=1.0 are 8–22 steps for nearly
every config — the truth's own correlation time is 8–9 steps, so phase
tracking cannot survive much past ~10 steps on a chaotic series (GyroSwin's
~110 vs 7–10 is a different metric on 5D states; qualitative context only).
The one apparent extension — ft oracle10 at 49 ID — is itself a level
effect: an under-dispersed forecast (r_σ ≈ 0.2) at the RIGHT level keeps
ε̃ ≈ 0.8 < τ without tracking anything. The invariant statistics show every
TSFM rollout under-disperses (r_σ ≤ 0.6 in all 26 rows; the r_σ<0.2
flatline fraction is 3/6–6/6 on ID even for the finetuned model; the
baselines are exact flatlines, τ_c = 1) and over-smooths (forecast τ_c
typically 12–17 vs truth 8–9). Example QUALITY, not finetuning, governs
what little structure survives: the oracle-fed ft model has the fewest
flatlines (3/6 ID, 1/5 OOD), the truth-closest ACF (d_acf 0.151), and a
truth-like τ_c (10) — copying the right level apparently comes with copying
truth-like fluctuation statistics.

(3) **Oracle gap: the pool always contains a near-twin; no function of the
80-step context can find it.** Pool-min d_lvl ≤ 0.9 for all 11 queries
(median 0.23) — label-aware selection has huge headroom, as the oracle
cells prove. But the oracle's picks sit at median ctx-rank 98/245 (chance
≈ 122) and pool-wide ρ(d_ctx, d_lvl) is only +0.25 — BY CONSTRUCTION:
every Phase-2 retrieval distance z-scores the context, erasing exactly the
level information the metric scores. Operating parameters do carry level
information (ρ(d_op, d_lvl) = +0.49, oracle op-rank 52/245), yet op_knn
scores ≈ random even on the param-CONDITIONED ft model (39.6 vs 40.0 ID;
the Phase-7 probe): partial level correlation does not put level-matched
examples into the top-k. The strongest context-side level signal is the
raw context MEAN (pool-wide ρ(ctx_mean, own tail) = +0.89 — the same
signal kNN-copy and shared scaling already exploit), but even a
ctx_mean-knn selector only reaches min d_lvl ≈ 5 ID (mmr k10: 3.1; oracle:
0.26) — an order of magnitude short. And best-pick level match alone is
NOT a sufficient statistic of the gain: the improvement-vs-min-d_lvl cloud
is only weakly monotone (Spearman −0.31, R² 0.20); context COMPOSITION —
the wrong-level example mass the Phase-6 win512 clamp removes — carries
the rest of the variance, which is why the labeled feature-knn
extrapolation promises no gain. The 80-step linear-phase context
under-determines the saturation level by ~5 flux units; closing the
oracle gap needs side information (a trained level predictor, more
physics, labels), not a better distance over the same 80 steps.

(4) **Per-trace: retrieval's RMSE gain is concentrated in the
highest-level traces.** ft k0's two big ID misses are the two ~146-level
traces (it_8 −39.1, it_262 −36.0); best-legit retrieval fixes exactly
those (+2.2, +15.9) while paying smaller new errors elsewhere (it_131
−1.3 → −23.1) — hence median per-trace improvement ≈ 0 while RMSE improves
22.2 → 15.6. OOD errors are systematic over-predictions on 4/5 traces that
legit retrieval barely moves (and ood_it_4's −59.6 under-prediction needs
the oracle to fix) — consistent with Phase 6's "OOD is finetuning's story
alone".
""".rstrip()


########################################################
# Self-test (CPU, synthetic + pool)
########################################################


def self_test() -> None:
    rng = np.random.default_rng(0)
    print("Mechanism-analysis self-tests (CPU)...")

    # T1 — decomposition identity, exact, on random series pairs
    for _ in range(25):
        y = rng.normal(3.0, 2.0, 266)
        f = rng.normal(2.5, 1.5, 266)
        d = decompose(y, f)
        assert abs(d["mse"] - (d["bias"] ** 2 + d["e_fluc"] ** 2)) <= 1e-10 * max(
            1.0, d["mse"]
        ), "T1: decomposition identity violated"
    y = rng.normal(5.0, 1.5, 266)
    d = decompose(y, np.full(266, 7.0))
    assert np.isclose(d["e_fluc"], float(y[-W:].std()), rtol=1e-12), (
        "T1: constant forecast must give e_fluc = std(y)"
    )
    assert d["r_sigma"] == 0.0, "T1: constant forecast must give r_sigma = 0"
    print("✓ T1: identity mse = b² + e_fluc² (atol 1e-10); constant forecast edge case")

    # T2 — ACF: matches an np.correlate reference exactly; AR(1) recovers rho
    x = rng.normal(size=80)
    xc = x - x.mean()
    full = np.correlate(xc, xc, "full")[len(x) - 1 :]
    reference = full[1 : ACF_PLOT_LAGS + 1] / full[0]
    assert np.allclose(acf(x, ACF_PLOT_LAGS), reference, atol=1e-12), (
        "T2: acf != np.correlate reference"
    )
    rho_true = 0.8
    n = 200_000
    ar = np.empty(n)
    ar[0] = 0.0
    noise = rng.normal(size=n)
    for t in range(1, n):
        ar[t] = rho_true * ar[t - 1] + noise[t]
    sample = acf(ar, 10)
    expected = rho_true ** np.arange(1, 11)
    assert np.max(np.abs(sample - expected)) < 0.02, (
        f"T2: AR(1) ACF off: {sample[:3]} vs {expected[:3]}"
    )
    tau_c = correlation_time(sample)
    assert tau_c == int(np.ceil(-1.0 / np.log(rho_true))), (
        f"T2: AR(1) correlation time {tau_c}"
    )
    assert np.array_equal(acf(np.full(80, 3.0), 5), np.zeros(5)), (
        "T2: flatline ACF must be zeros"
    )
    print("✓ T2: ACF ≡ np.correlate reference; AR(1) ρ=0.8 recovered; τ_c = 5")

    # T3 — horizon: perfect forecast censored; 3σ offset trips immediately;
    # monotone in τ
    y = rng.normal(0.0, 2.0, 266)
    eps_perfect = smoothed_tracking_error(y, y.copy())
    assert all(tracking_horizon(eps_perfect, tau) is None for tau in TAUS), (
        "T3: perfect forecast must be right-censored at every τ"
    )
    sigma = float(y[-W:].std())
    offset = y + 3.0 * sigma
    eps_off = smoothed_tracking_error(y, offset)
    h = tracking_horizon(eps_off, 1.0)
    assert h is not None and h <= SMOOTH_L, f"T3: 3σ offset H={h} > L={SMOOTH_L}"
    f_noisy = y + rng.normal(0.0, 1.5 * sigma, 266)
    eps_noisy = smoothed_tracking_error(y, f_noisy)
    hs = [tracking_horizon(eps_noisy, tau) for tau in TAUS]
    filled = [HORIZON_MAX if h is None else h for h in hs]
    assert filled == sorted(filled), f"T3: horizons not monotone in τ: {hs}"
    print("✓ T3: perfect→censored; 3σ-offset→H≤L; τ-monotonicity")

    # T4 — oracle d_lvl = pool minimum (real pool + selection)
    from fusiontimeseries.benchmarking.few_shot.selection import (
        _benchmark_true_tail_means,
        select_examples_oracle,
    )

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    pool_tails = np.array([float(np.mean(ex.trace_array[-W:])) for ex in pool])
    true_means = _benchmark_true_tail_means(W)
    for key, true_mean in true_means.items():
        top = select_examples_oracle(pool, 1, key)[0]
        d_top = abs(float(np.mean(top.trace_array[-W:])) - true_mean)
        assert d_top == float(np.min(np.abs(pool_tails - true_mean))), (
            f"T4: oracle top-pick d_lvl != pool minimum for {key}"
        )
    print(f"✓ T4: oracle top-pick d_lvl = pool minimum for all {len(true_means)} queries")

    print("\n✅ Mechanism-analysis self-tests passed!")


########################################################
# Report assembly
########################################################


def make_header(cells: dict[str, dict]) -> list[str]:
    n_bit = sum(c["consistency"]["n_bit_equal"] for c in cells.values())
    n_cmp = sum(c["consistency"]["n_compared"] for c in cells.values())
    return [
        "# Mechanism analysis — Phase 7 (where do the gains come from?)",
        "",
        "Phases 2–6 measured THAT retrieval-ICL and finetuning help; this "
        "analysis explains WHY. Data: full-forecast dumps of 13 headline "
        "cells (`results/few_shot_v7_mechanism/`, re-run through the frozen "
        "harness with a forecast-capture hook — shared scaling + mean "
        f"decoding throughout) whose per-trace tails reproduced the recorded "
        f"v5/v6 scalars bit-exactly ({n_bit}/{n_cmp} reference comparisons "
        "bit-equal at dump time; re-asserted from the stored JSONs at every "
        "analyzer load), plus the Phase-6 scalar grid for the oracle-gap "
        "block (no model runs there). Framing: the context-parroting "
        "literature ([2505.11349](https://arxiv.org/abs/2505.11349), "
        "[2409.15771](https://arxiv.org/abs/2409.15771)) — TSFMs forecast "
        "chaotic systems by copying context motifs, fail pointwise quickly, "
        "yet preserve invariant statistics; our benchmark metric (tail-mean "
        "RMSE) is exactly such an invariant statistic.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-7 mechanism analysis")
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--v6-dir", type=Path, default=DEFAULT_V6_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the CPU self-tests and exit (writes nothing)",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    cells = load_dump_cells(args.dump_dir)
    print(f"✓ {len(cells)} dump cells loaded; stored forecasts re-asserted")
    v6_index = build_reference_index(args.v6_dir)
    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    assert len(pool) == 245, f"Expected fixed pool of 245, got {len(pool)}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = make_header(cells)

    decomp_rows, decomp_points = collect_decomposition(cells)
    lines.extend(make_decomposition_block(decomp_rows, decomposition_claim_rows(decomp_rows)))
    decomp_path = fig_decomposition(decomp_rows, decomp_points, args.out_dir)
    lines.append(f"\n![{decomp_path.stem}]({decomp_path.name})")

    horizon_rows, eps_curves = collect_horizons(cells)
    acf_rows, acf_curves, truth_curves, truth_info = collect_acf_stats(cells)
    lines.extend(make_horizon_block(horizon_rows, acf_rows, truth_info))
    horizon_path = fig_horizon(horizon_rows, eps_curves, args.out_dir)
    acf_path = fig_acf(acf_curves, truth_curves, args.out_dir)
    lines.append(f"\n![{horizon_path.stem}]({horizon_path.name})")
    lines.append(f"\n![{acf_path.stem}]({acf_path.name})")

    truths = cells["ft k0"]["_truths"]
    gap = analyze_oracle_gap(v6_index, pool, truths)
    lines.extend(make_gap_block(gap))
    gap_path = fig_oracle_gap(gap, args.out_dir)
    lines.append(f"\n![{gap_path.stem}]({gap_path.name})")

    lines.extend(make_per_trace_block(gap, v6_index))

    grid_paths = fig_forecast_grids(cells, args.out_dir)
    lines += [
        "\n## 5. Forecast grids",
        "",
        "Truth + forecast overlays for the 7 headline configs (seed 42, "
        "vertical line = forecast start):",
        "",
    ]
    for path in grid_paths:
        lines.append(f"- `{path.name}`")

    lines.append("\n## Verdict\n")
    lines.append(VERDICT)

    markdown = "\n".join(lines) + "\n"
    table_path = args.out_dir / "mechanism_table.md"
    table_path.write_text(markdown)
    print(markdown)
    for path in (table_path, decomp_path, horizon_path, acf_path, gap_path, *grid_paths):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
