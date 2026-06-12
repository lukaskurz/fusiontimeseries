"""Analysis of the Phase-4 covariate grid: tables, figures, significance.

Loads ``results/few_shot_v4_covariates/`` and produces, in
``docs/results/fewshot/``:

- ``covariates_table.md`` — anchors (zeroshot ± constant channels),
  per-strategy k-tables with no-cov / +cov / +perm interleaved, the group
  block, the channel-contrast diagnostic, paired comparisons, the full
  adaptation-ladder table, and interpretation notes;
- ``covariates_kcurves.png`` — per-strategy k-curves, +cov solid vs no-cov
  faded vs permuted dotted;
- ``covariates_contrast_scatter.png`` — per-trace channel contrast vs the
  +cov effect on that trace (op_knn = the low-contrast pole);
- ``adaptation_ladder.png`` — two-panel ID/OOD bars ordered by adaptation
  cost (the bridge deliverable).

Method labels are parsed as ``{slug with / -> _}_{strategy}__{variant}``
with the v4 token set (``shared`` | ``group``; ``opcov`` | ``permcov``) and
cross-checked against the FewShotConfig metadata. The analyzer hard-asserts
per-seed example-id equality for every +cov / +perm / group+cov cell against
its no-cov twin.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_covariates
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fusiontimeseries.benchmarking.few_shot.harness import (
    PairedComparison,
    load_results,
    paired_comparison,
)
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    OP_NAMES,
    get_params_for_benchmark_trace,
    get_params_for_raw_id,
    normalize_params,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import MODEL_SLUGS

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v4_covariates"
DEFAULT_OUT_DIR: Path = REPO_ROOT / "docs" / "results" / "fewshot"

CHRONOS2_SLUG = MODEL_SLUGS["chronos2"]
SPLITS: tuple[str, ...] = ("in_distribution", "out_of_distribution")
SPLIT_SHORT: dict[str, str] = {"in_distribution": "ID", "out_of_distribution": "OOD"}
MAIN_STRATEGIES: tuple[str, ...] = (
    "random",
    "op_knn",
    "ctx_euclid",
    "mmr_euclid",
    "oracle_tail",
)
LEGIT_EXCLUDED: tuple[str, ...] = ("oracle_tail", "zeroshot")

#: Index key: (slug, strategy, k, presentation, op_covariates)
Key = tuple[str, str, int, str, str | None]

#: Segment lengths of the t266 concat stream (full-length example targets).
EXAMPLE_SEGMENT_LENGTH: int = 267
QUERY_SEGMENT_LENGTH: int = 80

STRATEGY_STYLE: dict[str, dict] = {
    "random": {"color": "0.35", "marker": "o"},
    "op_knn": {"color": "tab:blue", "marker": "s"},
    "ctx_euclid": {"color": "tab:orange", "marker": "^"},
    "mmr_euclid": {"color": "tab:brown", "marker": "P"},
    "oracle_tail": {"color": "tab:red", "marker": "*"},
}

#: Cross-run / cross-pipeline reference points for the adaptation ladder.
#: These are NOT from the v4 grid run — provenance per entry; treat as
#: context, not cell-to-cell comparable numbers.
REFERENCE_LADDER: list[dict] = [
    {
        # GyroSwin paper (arXiv:2510.07314), Table: GPR baseline on the same
        # 6 ID / 5 OOD heat-flux benchmark traces.
        "label": "GPR (paper baseline)",
        "id": 43.82,
        "ood": 59.28,
        "kind": "reference",
        "cost": "classical surrogate",
    },
    {
        # GyroSwin paper (arXiv:2510.07314): the full 5D surrogate.
        "label": "GyroSwin-1B (paper)",
        "id": 18.35,
        "ood": 26.43,
        "kind": "reference",
        "cost": "full surrogate training",
    },
    # METRIC CAVEAT (found in Phase 6): the chronos2 finetuning notebooks
    # score mean(x[:-80]) — the mean over everything EXCEPT the tail,
    # INCLUDING the 80 copied ground-truth context steps — while every other
    # rung here is tail RMSE (mean(x[-80:])). These three rows are therefore
    # NOT metric-comparable to the rest of the ladder; the honest finetuned
    # rungs are measured in Phase 6 (analyze_finetuned.py, which regenerates
    # adaptation_ladder.png with v6 cells).
    {
        # Severin's finetuning runs (this repo, README finetuning section):
        # Chronos-2 + BilinearLoRA with operating-param conditioning.
        "label": "Chronos-2 BilinearLoRA (finetuned; [:-80] metric)",
        "id": 13.83,
        "ood": 4.86,
        "kind": "reference",
        "cost": "finetuning",
    },
    {
        # Severin's finetuning runs: OSSBilinearLoRA variant.
        "label": "Chronos-2 OSSBilinearLoRA (finetuned; [:-80] metric)",
        "id": 16.11,
        "ood": 3.19,
        "kind": "reference",
        "cost": "finetuning",
    },
    {
        # Severin's finetuning runs: full finetune.
        "label": "Chronos-2 full FT ([:-80] metric)",
        "id": 15.50,
        "ood": 4.76,
        "kind": "reference",
        "cost": "finetuning",
    },
]

INTERPRETATION_NOTES = """
## Interpretation notes

- **Why step channels (and not constant covariates).** Chronos-2
  instance-norms every variate row INDEPENDENTLY (loc=nanmean, scale=RMS,
  + arcsinh; positive-affine invariant), so only within-row relative
  geometry survives. A constant channel has its VALUE erased exactly —
  smoke S1a showed two different param values engineered to share the
  post-norm rows give bit-identical forecasts — surviving only as a
  float32 tri-state rounding artifact (all-0/±1) that nonetheless shifts
  the rollout (~16% rel L2 at k=0): constant channels inject
  value-uncorrelated perturbation, not information. That is what the
  zeroshot+cov anchor row measures. Static params can therefore act ONLY
  via within-row contrast: example i's params over its segment, the
  query's over the query segment.
- **Relative-only ceiling.** Absolute parameter values cannot survive the
  per-row norm under any encoding — this is training-free *relative*
  conditioning. At k=1 the step degenerates to one sign bit per param
  (two values over ~equal supports normalize to ±1 regardless of the
  values) — the k=1 diagnostic block quantifies that floor.
- **op_knn is structurally confounded with the channels.** Selecting
  examples whose params are close to the query's re-flattens the step
  channel toward a constant (the erased case). The channel-contrast
  column makes this visible: op_knn sits at the low-contrast pole, so
  "op_knn + cov" cannot separate selection-conditioning from
  covariate-conditioning. Use random/ctx_euclid +cov for attribution.
- **Permuted-params control.** +cov differing from no-cov only shows that
  extra rows change the model's behaviour; +cov separating from +perm
  (same rows, example params shuffled, query true) is what attributes a
  gain to parameter INFORMATION. A +cov gain without permcov separation
  is presentation noise.
- **Group+cov is structurally degenerate by construction**: group mode has
  no per-example slot (each example IS a row), and the added per-task
  constant channels are value-erased as above. The group block is the
  empirical row for that claim.
- **Context clamp.** At k=10 the concat stream is ~2750 steps; Chronos-2
  left-clamps ALL rows (target + channels, one index range — alignment
  safe) to 2048, so k=10 cells partially measure the context window, the
  same caveat class as Phase 3.
- MPS is not bit-deterministic across process runs: headline comparisons
  live within the single v4 grid run; ladder reference rows are cross-run
  context with provenance noted.
- Wilcoxon p floors at 0.031 (n=6 ID) / 0.0625 (n=5 OOD) under
  pairing="trace"; the bootstrap CI is the primary evidence. Where seed
  sets are identical multi-seed, a pairing="trace_seed" row adds
  resolution but treats seeds as independent.
""".strip()


########################################################
# Parsing and indexing
########################################################


def parse_variant(variant: str) -> dict | None:
    """Decode a v4 hyphen-joined variant label into config fields."""
    fields: dict = {
        "presentation": "concat",
        "normalization": "per_example",
        "op_covariates": None,
    }
    if variant == "base":
        return fields
    for token in variant.split("-"):
        if token == "group":
            fields["presentation"] = "group"
        elif token == "shared":
            fields["normalization"] = "shared"
        elif token == "opcov":
            fields["op_covariates"] = "step"
        elif token == "permcov":
            fields["op_covariates"] = "permuted"
        else:
            return None
    return fields


def build_index(results: list[dict]) -> dict[Key, dict]:
    """Index v4 results; latest timestamp wins; tokens cross-checked."""
    index: dict[Key, dict] = {}
    for r in results:
        config = r.get("config", {})
        slug = config.get("model_slug")
        if slug is None or slug not in MODEL_SLUGS.values():
            continue
        prefix = slug.replace("/", "_") + "_"
        method = r.get("method", "")
        if not method.startswith(prefix) or "__" not in method:
            continue
        strategy, variant = method[len(prefix) :].rsplit("__", 1)
        fields = parse_variant(variant)
        if fields is None:
            print(f"WARNING: unparseable variant {variant!r} in {method}")
            continue
        for name, value in fields.items():
            recorded = config.get(name, value)
            assert recorded == value, (
                f"{method}: variant token says {name}={value!r} but config "
                f"records {recorded!r}"
            )
        key: Key = (
            slug,
            strategy,
            int(config["k_shot"]),
            fields["presentation"],
            fields["op_covariates"],
        )
        if key not in index or r["timestamp"] > index[key]["timestamp"]:
            index[key] = r
    return index


def get(
    index: dict[Key, dict],
    strategy: str,
    k: int,
    presentation: str = "concat",
    op_covariates: str | None = None,
) -> dict | None:
    return index.get((CHRONOS2_SLUG, strategy, k, presentation, op_covariates))


def rmse_of(result: dict, split: str) -> float:
    return float(result[split]["rmse"])


def std_of(result: dict, split: str) -> float | None:
    value = result[split].get("rmse_std_seeds")
    return None if value is None else float(value)


def assert_cov_twin_example_ids(index: dict[Key, dict]) -> int:
    """Hard-assert per-seed example-id equality vs the no-cov twin."""
    checked = 0
    for key, cov_result in index.items():
        slug, strategy, k, presentation, op_covariates = key
        if op_covariates is None or strategy == "zeroshot":
            continue
        twin = index.get((slug, strategy, k, presentation, None))
        assert twin is not None, f"Missing no-cov twin for {key}"
        assert cov_result["seeds"] == twin["seeds"], f"Seed mismatch for {key}"
        for sc, st in zip(cov_result["per_seed"], twin["per_seed"]):
            assert sc["example_ids"] == st["example_ids"], (
                f"example_ids differ for {key} seed {sc['seed']}"
            )
        checked += 1
    return checked


########################################################
# Channel-contrast diagnostic
########################################################


def _weighted_std(values: list[float], weights: list[int]) -> float:
    """Std over finite (value, weight) pairs; NaN segments are masked."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v)
    v, w = v[mask], w[mask]
    if v.size == 0:
        return float("nan")
    mean = float(np.average(v, weights=w))
    return float(np.sqrt(np.average((v - mean) ** 2, weights=w)))


def channel_contrast(result: dict) -> dict[str, float]:
    """Per-trace within-stream std of the normalized param channel values.

    Built from the recorded ``example_ids`` (stream order) plus the query's
    own params, weighted by segment lengths (267 per example, 80 for the
    initial query block); averaged over the 4 params and over seeds. Low
    contrast ⇒ the channels are near-constant ⇒ near the erased case.
    """
    per_trace: dict[str, list[float]] = {}
    for seed_result in result["per_seed"]:
        for trace_key, example_ids in seed_result["example_ids"].items():
            query_norm = normalize_params(get_params_for_benchmark_trace(trace_key))
            param_stds = []
            for name in OP_NAMES:
                values, weights = [], []
                for raw_id in example_ids:
                    params = get_params_for_raw_id(raw_id)
                    values.append(
                        normalize_params(params)[name] if params is not None else float("nan")
                    )
                    weights.append(EXAMPLE_SEGMENT_LENGTH)
                values.append(query_norm[name])
                weights.append(QUERY_SEGMENT_LENGTH)
                param_stds.append(_weighted_std(values, weights))
            per_trace.setdefault(trace_key, []).append(float(np.nanmean(param_stds)))
    return {trace_key: float(np.mean(stds)) for trace_key, stds in per_trace.items()}


def per_trace_abs_errors(result: dict) -> dict[str, float]:
    """Seed-averaged |tail error| per trace."""
    by_trace: dict[str, list[float]] = {}
    for seed_result in result["per_seed"]:
        for tr in seed_result["per_trace"]:
            by_trace.setdefault(tr["trace_key"], []).append(tr["abs_error"])
    return {trace_key: float(np.mean(v)) for trace_key, v in by_trace.items()}


########################################################
# Table
########################################################


def fmt_cell(result: dict | None) -> str:
    if result is None:
        return "—"
    parts = []
    for split in SPLITS:
        rmse = rmse_of(result, split)
        std = std_of(result, split)
        parts.append(f"{rmse:.2f}±{std:.2f}" if std is not None else f"{rmse:.2f}")
    return " / ".join(parts)


def table_block(
    index: dict[Key, dict], rows: list[tuple[str, str, dict]], ks: list[int]
) -> list[str]:
    lines = ["| variant | " + " | ".join(f"k={k}" for k in ks) + " |"]
    lines.append("|" + "---|" * (len(ks) + 1))
    for label, strategy, variant in rows:
        cells = [fmt_cell(get(index, strategy, k, **variant)) for k in ks]
        if all(cell == "—" for cell in cells):
            continue
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def best_legit(
    index: dict[Key, dict], op_covariates: str | None
) -> tuple[Key, dict] | None:
    """Minimal-ID-RMSE legitimate concat cell for a covariate mode."""
    candidates = [
        (key, r)
        for key, r in index.items()
        if key[3] == "concat"
        and key[4] == op_covariates
        and key[1] not in LEGIT_EXCLUDED
        and key[2] > 0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda kr: rmse_of(kr[1], "in_distribution"))


def describe_key(key: Key) -> str:
    _, strategy, k, presentation, op_covariates = key
    tokens = [strategy, f"k={k}"]
    if presentation != "concat":
        tokens.append(presentation)
    if op_covariates == "step":
        tokens.append("+cov")
    elif op_covariates == "permuted":
        tokens.append("+perm")
    return " ".join(tokens)


def make_table(index: dict[Key, dict]) -> list[str]:
    lines = ["# Training-free operating-parameter conditioning — Phase 4 results", ""]
    lines.append(
        "Chronos-2 only (the only benchmarked model with zero-shot covariate "
        "support). Protocol: fixed 245-trace pool, full-length example "
        "targets (t266), context 80, prediction 64, tail 80, flat concat + "
        "SHARED scaling (the Phase-3 winner) for every concat cell. `+cov` "
        "adds the 4 operating parameters (q, ŝ, R/L_T, R/L_n) as "
        "step-function covariate channels over the ICL stream; `+perm` is "
        "the permuted-params control (example params shuffled per example "
        "set, query true). Cells are `ID / OOD` tail RMSE; multi-seed cells "
        "mean±std (random: 20 seeds, group random: 5), deterministic cells "
        "a single pass (seed 42). All cells from ONE grid run "
        "(`results/few_shot_v4_covariates/`)."
    )

    # Anchors
    z = get(index, "zeroshot", 0)
    zc = get(index, "zeroshot", 0, op_covariates="step")
    if z is not None and zc is not None:
        lines.append("\n## Anchors — constant channels are degenerate\n")
        lines.append(
            "At k=0 the channels are CONSTANT rows: the per-row instance "
            "norm erases the parameter values exactly (smoke S1a), leaving "
            "only a float32 rounding tri-state plus the presence of 8 extra "
            "rows. The offset below therefore measures presentation "
            "perturbation, not conditioning:\n"
        )
        lines.append("| anchor | ID | OOD |")
        lines.append("|---|---|---|")
        lines.append(
            f"| zero-shot (k=0) | {rmse_of(z, 'in_distribution'):.2f} | "
            f"{rmse_of(z, 'out_of_distribution'):.2f} |"
        )
        lines.append(
            f"| zero-shot + constant op channels | {rmse_of(zc, 'in_distribution'):.2f} | "
            f"{rmse_of(zc, 'out_of_distribution'):.2f} |"
        )

    # Per-strategy k tables
    ks = sorted({key[2] for key in index if key[3] == "concat" and key[2] > 0})
    lines.append("\n## Step-channel conditioning by strategy (concat, shared scaling)\n")
    lines.append(
        "k=1 (where present) is the sign-bit degeneracy diagnostic: the "
        "step carries one bit per param. op_knn structurally flattens its "
        "own channels (see the contrast column and notes).\n"
    )
    for strategy in MAIN_STRATEGIES:
        rows = [
            (f"{strategy}", strategy, {}),
            (f"{strategy} +cov", strategy, {"op_covariates": "step"}),
            (f"{strategy} +perm", strategy, {"op_covariates": "permuted"}),
        ]
        block = table_block(index, rows, ks)
        if len(block) > 2:
            lines.append(f"\n### {strategy}\n")
            lines.extend(block)

    # Group block
    group_ks = sorted({key[2] for key in index if key[3] == "group" and key[2] > 0})
    if group_ks:
        lines.append("\n## Group ICL + constant channels (structural inertness row)\n")
        lines.append(
            "Group mode has no per-example parameter slot; the added "
            "channels are per-task constants → value-erased. Identical "
            "example sets per pair (hard-asserted); random uses 5 seeds "
            "here.\n"
        )
        rows = []
        for strategy in ("random", "ctx_euclid"):
            rows.append((f"{strategy} (group)", strategy, {"presentation": "group"}))
            rows.append(
                (
                    f"{strategy} (group +cov)",
                    strategy,
                    {"presentation": "group", "op_covariates": "step"},
                )
            )
        lines.extend(table_block(index, rows, group_ks))

    # Channel-contrast diagnostic
    lines.append("\n## Channel-contrast diagnostic (+cov cells)\n")
    lines.append(
        "Within-stream weighted std of the normalized param channel values "
        "(mean over the 4 params, seeds, traces). The per-row norm erases "
        "everything but within-row contrast, so low contrast ⇒ the channel "
        "approaches the erased constant case. op_knn selects examples with "
        "params ≈ the query's — flattening its own channels.\n"
    )
    lines.append("| strategy | " + " | ".join(f"k={k}" for k in ks) + " |")
    lines.append("|" + "---|" * (len(ks) + 1))
    for strategy in MAIN_STRATEGIES:
        cells = []
        for k in ks:
            r = get(index, strategy, k, op_covariates="step")
            if r is None:
                cells.append("—")
            else:
                contrast = channel_contrast(r)
                cells.append(f"{np.mean(list(contrast.values())):.3f}")
        if all(cell == "—" for cell in cells):
            continue
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")

    return lines


########################################################
# Figures
########################################################


def fig_kcurves(index: dict[Key, dict], out_dir: Path) -> Path | None:
    ks = sorted({key[2] for key in index if key[3] == "concat" and key[2] > 0})
    if not ks:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    modes = (
        (None, 0.3, 1.3, "--", ""),
        ("step", 1.0, 2.0, "-", " +cov"),
        ("permuted", 0.65, 1.5, ":", " +perm"),
    )
    for ax, split in zip(axes, SPLITS):
        for strategy in MAIN_STRATEGIES:
            style = STRATEGY_STYLE[strategy]
            for op_covariates, alpha, lw, ls, suffix in modes:
                points = [
                    (k, r)
                    for k in ks
                    if (r := get(index, strategy, k, op_covariates=op_covariates))
                ]
                if not points:
                    continue
                xs = [k for k, _ in points]
                ys = [rmse_of(r, split) for _, r in points]
                ax.plot(
                    xs, ys, color=style["color"], marker=style["marker"],
                    linestyle=ls, linewidth=lw, alpha=alpha, markersize=5,
                    label=f"{strategy}{suffix}",
                )
                if strategy == "random" and op_covariates in (None, "step"):
                    stds = [std_of(r, split) or 0.0 for _, r in points]
                    ax.fill_between(
                        xs,
                        [y - s for y, s in zip(ys, stds)],
                        [y + s for y, s in zip(ys, stds)],
                        color=style["color"], alpha=0.10 * alpha + 0.05,
                    )
        for op_covariates, color, label in (
            (None, "0.6", "zero-shot"),
            ("step", "tab:cyan", "zero-shot + const channels"),
        ):
            anchor = get(index, "zeroshot", 0, op_covariates=op_covariates)
            if anchor is not None:
                ax.axhline(
                    rmse_of(anchor, split), color=color, linestyle=":",
                    linewidth=1.2, label=label,
                )
        ax.set_xticks(ks)
        ax.set_xlabel("k (examples)")
        ax.grid(alpha=0.3)
        ax.set_title(f"Chronos-2 — {SPLIT_SHORT[split]}", fontsize=11)
    axes[0].set_ylabel("tail RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, label in zip(handles, labels):
        seen.setdefault(label, h)
    fig.legend(
        list(seen.values()), list(seen), loc="lower center", ncol=5,
        fontsize=7.5, frameon=False,
    )
    fig.suptitle(
        "Operating-param step channels: +cov (solid) vs no-cov (faded) vs permuted (dotted), "
        "shared scaling",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    path = out_dir / "covariates_kcurves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_contrast_scatter(index: dict[Key, dict], out_dir: Path) -> Path | None:
    """Per-trace channel contrast vs the +cov effect on |tail error|."""
    points: dict[str, tuple[list[float], list[float]]] = {}
    for key, cov_result in index.items():
        _, strategy, k, presentation, op_covariates = key
        if op_covariates != "step" or presentation != "concat" or k == 0:
            continue
        twin = index.get((CHRONOS2_SLUG, strategy, k, "concat", None))
        if twin is None:
            continue
        contrast = channel_contrast(cov_result)
        err_cov = per_trace_abs_errors(cov_result)
        err_twin = per_trace_abs_errors(twin)
        xs, ys = points.setdefault(strategy, ([], []))
        for trace_key in contrast:
            xs.append(contrast[trace_key])
            ys.append(err_cov[trace_key] - err_twin[trace_key])
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for strategy, (xs, ys) in points.items():
        style = STRATEGY_STYLE[strategy]
        ax.scatter(
            xs, ys, s=22, alpha=0.7, color=style["color"],
            marker=style["marker"], label=strategy,
        )
    ax.axhline(0.0, color="0.4", linewidth=1.0)
    ax.set_xlabel("channel contrast (within-stream std of normalized params)")
    ax.set_ylabel("Δ|tail error| (+cov − no-cov), per trace")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Low contrast ⇒ channels approach the erased constant case", fontsize=10)
    fig.tight_layout()
    path = out_dir / "covariates_contrast_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def ladder_rows(index: dict[Key, dict]) -> list[dict]:
    """Adaptation ladder: measured v4 rows + hardcoded references."""
    rows: list[dict] = []
    z = get(index, "zeroshot", 0)
    if z is not None:
        rows.append(
            {
                "label": "Chronos-2 zero-shot",
                "id": rmse_of(z, "in_distribution"),
                "ood": rmse_of(z, "out_of_distribution"),
                "kind": "v4",
                "cost": "none",
            }
        )
    icl = best_legit(index, None)
    if icl is not None:
        rows.append(
            {
                "label": f"Chronos-2 ICL ({describe_key(icl[0])})",
                "id": rmse_of(icl[1], "in_distribution"),
                "ood": rmse_of(icl[1], "out_of_distribution"),
                "kind": "v4",
                "cost": "in-context examples",
            }
        )
    icl_cov = best_legit(index, "step")
    if icl_cov is not None:
        rows.append(
            {
                "label": f"Chronos-2 ICL + OP covariates ({describe_key(icl_cov[0])})",
                "id": rmse_of(icl_cov[1], "in_distribution"),
                "ood": rmse_of(icl_cov[1], "out_of_distribution"),
                "kind": "v4",
                "cost": "+ op-param channels",
            }
        )
    oracle = min(
        (
            (key, r)
            for key, r in index.items()
            if key[1] == "oracle_tail" and key[3] == "concat"
        ),
        key=lambda kr: rmse_of(kr[1], "in_distribution"),
        default=None,
    )
    if oracle is not None:
        rows.append(
            {
                "label": f"oracle_tail ceiling ({describe_key(oracle[0])}, cheats)",
                "id": rmse_of(oracle[1], "in_distribution"),
                "ood": rmse_of(oracle[1], "out_of_distribution"),
                "kind": "oracle",
                "cost": "label-aware (diagnostic)",
            }
        )
    # Ladder order: classical baseline, then ascending adaptation cost,
    # references interleaved where they belong.
    ordered = [REFERENCE_LADDER[0]]  # GPR
    ordered += [row for row in rows if row["kind"] in ("v4", "oracle")]
    ordered += REFERENCE_LADDER[2:]  # finetuned chronos2 rows
    ordered.append(REFERENCE_LADDER[1])  # GyroSwin-1B
    return ordered


def fig_ladder(rows: list[dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors = {"reference": "0.65", "v4": "tab:blue", "oracle": "tab:red"}
    for ax, split_key, title in ((axes[0], "id", "ID"), (axes[1], "ood", "OOD")):
        labels = [row["label"] for row in rows]
        values = [row[split_key] for row in rows]
        bar_colors = [colors[row["kind"]] for row in rows]
        hatches = ["//" if row["kind"] == "oracle" else "" for row in rows]
        y = np.arange(len(rows))
        bars = ax.barh(y, values, color=bar_colors, height=0.62)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for yi, value in zip(y, values):
            ax.text(value + 0.5, yi, f"{value:.1f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("tail RMSE")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
        plt.Rectangle((0, 0), 1, 1, color="tab:red", hatch="//"),
        plt.Rectangle((0, 0), 1, 1, color="0.65"),
    ]
    fig.legend(
        handles,
        ["this grid run (v4)", "cheating diagnostic", "cross-run reference"],
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )
    fig.suptitle(
        "Adaptation ladder: zero-shot → ICL → ICL + OP covariates → finetuned → full surrogate",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    path = out_dir / "adaptation_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


########################################################
# Paired comparisons
########################################################


def fmt_comparison(label: str, comp: PairedComparison) -> str:
    wilcoxon = "n/a" if comp.wilcoxon_p is None else f"{comp.wilcoxon_p:.3f}"
    return (
        f"| {label} | {SPLIT_SHORT[comp.split]} | {comp.rmse_a:.2f} | {comp.rmse_b:.2f} | "
        f"{comp.rmse_diff:+.2f} | [{comp.bootstrap_ci_low:.2f}, {comp.bootstrap_ci_high:.2f}] | "
        f"{comp.bootstrap_p:.3f} | {wilcoxon} |"
    )


def comparison_rows(label: str, a: dict, b: dict) -> list[str]:
    rows = []
    same_seeds = a["seeds"] == b["seeds"] and len(a["seeds"]) > 1
    for split in SPLITS:
        rows.append(fmt_comparison(label, paired_comparison(a, b, split, pairing="trace")))
        if same_seeds:
            rows.append(
                fmt_comparison(
                    f"{label} [trace_seed]",
                    paired_comparison(a, b, split, pairing="trace_seed"),
                )
            )
    return rows


def make_comparisons(index: dict[Key, dict]) -> list[str]:
    lines = [
        "\n## Paired comparisons (pairing=trace, bootstrap CI primary)",
        "",
        "Δ = RMSE(A) − RMSE(B); negative favours A. CI/p from a 10k paired "
        "bootstrap over traces; Wilcoxon shown for completeness (p floors "
        "0.031 ID / 0.0625 OOD). `[trace_seed]` rows appear where seed sets "
        "are identical.",
    ]
    header = (
        "\n| A vs B | split | RMSE A | RMSE B | Δ(A−B) | 95% CI | p_boot | p_wilcoxon |\n"
        "|---|---|---|---|---|---|---|---|"
    )

    ks = sorted({key[2] for key in index if key[3] == "concat" and key[2] > 0})

    # Anchor offset
    z = get(index, "zeroshot", 0)
    zc = get(index, "zeroshot", 0, op_covariates="step")
    if z is not None and zc is not None:
        lines.append("\n### Anchor — constant channels (row presence + tri-state artifact)")
        lines.append(header)
        lines.extend(comparison_rows("zeroshot+cov vs zeroshot", zc, z))

    # +cov vs no-cov per (strategy, k)
    lines.append("\n### Step channels: +cov vs no-cov")
    lines.append(header)
    for strategy in MAIN_STRATEGIES:
        for k in ks:
            cov = get(index, strategy, k, op_covariates="step")
            nocov = get(index, strategy, k)
            if cov is not None and nocov is not None:
                lines.extend(comparison_rows(f"{strategy}+cov(k={k}) vs {strategy}(k={k})", cov, nocov))

    # Permutation control
    perm_rows: list[str] = []
    for strategy in ("random", "ctx_euclid"):
        for k in ks:
            perm = get(index, strategy, k, op_covariates="permuted")
            if perm is None:
                continue
            cov = get(index, strategy, k, op_covariates="step")
            nocov = get(index, strategy, k)
            if cov is not None:
                perm_rows.extend(
                    comparison_rows(f"{strategy}+cov(k={k}) vs {strategy}+perm(k={k})", cov, perm)
                )
            if nocov is not None:
                perm_rows.extend(
                    comparison_rows(f"{strategy}+perm(k={k}) vs {strategy}(k={k})", perm, nocov)
                )
    if perm_rows:
        lines.append("\n### Attribution — permuted-params control")
        lines.append(header)
        lines.extend(perm_rows)

    # Selection- vs covariate-conditioning
    sel_rows: list[str] = []
    for k in ks:
        op_nocov = get(index, "op_knn", k)
        rand_cov = get(index, "random", k, op_covariates="step")
        if op_nocov is not None and rand_cov is not None:
            sel_rows.extend(
                comparison_rows(f"op_knn(k={k}) vs random+cov(k={k})", op_nocov, rand_cov)
            )
    if sel_rows:
        lines.append("\n### Selection-conditioning vs covariate-conditioning")
        lines.append(header)
        lines.extend(sel_rows)

    # Group block
    group_ks = sorted({key[2] for key in index if key[3] == "group" and key[2] > 0})
    group_rows: list[str] = []
    for strategy in ("random", "ctx_euclid"):
        for k in group_ks:
            gcov = get(index, strategy, k, presentation="group", op_covariates="step")
            g = get(index, strategy, k, presentation="group")
            if gcov is not None and g is not None:
                group_rows.extend(
                    comparison_rows(f"{strategy} group+cov(k={k}) vs group(k={k})", gcov, g)
                )
    if group_rows:
        lines.append("\n### Group block — constant channels in group mode")
        lines.append(header)
        lines.extend(group_rows)

    # Headline rows
    lines.append("\n### Headline")
    lines.append(header)
    icl = best_legit(index, None)
    icl_cov = best_legit(index, "step")
    if icl is not None and icl_cov is not None:
        lines.extend(
            comparison_rows(
                f"best legit +cov [{describe_key(icl_cov[0])}] vs best legit no-cov "
                f"[{describe_key(icl[0])}]",
                icl_cov[1],
                icl[1],
            )
        )
    if icl is not None and z is not None:
        best_overall = icl if icl_cov is None or rmse_of(icl[1], "in_distribution") <= rmse_of(
            icl_cov[1], "in_distribution"
        ) else icl_cov
        lines.extend(
            comparison_rows(
                f"best legit v4 [{describe_key(best_overall[0])}] vs zero-shot",
                best_overall[1],
                z,
            )
        )
    return lines


def make_ladder_table(rows: list[dict]) -> list[str]:
    lines = [
        "\n## Adaptation ladder (the bridge table)",
        "",
        "Ordered by adaptation cost. v4 rows come from this grid run; "
        "reference rows are cross-run/cross-pipeline context (GyroSwin "
        "paper arXiv:2510.07314; Severin's finetuning results in this "
        "repo's README) — same 11 benchmark traces and tail-RMSE metric, "
        "but not cell-to-cell comparable with v4 (different runs).",
        "",
        "| rung | adaptation cost | ID | OOD | source |",
        "|---|---|---|---|---|",
    ]
    source = {"v4": "this run", "oracle": "this run (cheats)", "reference": "cross-run reference"}
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['cost']} | {row['id']:.2f} | {row['ood']:.2f} | "
            f"{source[row['kind']]} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Phase-4 covariates grid")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    index = build_index(load_results(args.results_dir))
    assert index, f"No covariate-grid results in {args.results_dir}"
    n_pairs = assert_cov_twin_example_ids(index)
    print(f"✓ example_ids identical for all {n_pairs} cov/no-cov twin pairs")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = make_table(index)
    figure_paths = []
    for fig_path in (
        fig_kcurves(index, args.out_dir),
        fig_contrast_scatter(index, args.out_dir),
    ):
        if fig_path is not None:
            figure_paths.append(fig_path)
            lines.append(f"\n![{fig_path.stem}]({fig_path.name})")
    lines.extend(make_comparisons(index))
    rows = ladder_rows(index)
    lines.extend(make_ladder_table(rows))
    ladder_path = fig_ladder(rows, args.out_dir)
    figure_paths.append(ladder_path)
    lines.append(f"\n![{ladder_path.stem}]({ladder_path.name})")
    lines.append("\n" + INTERPRETATION_NOTES)

    markdown = "\n".join(lines) + "\n"
    table_path = args.out_dir / "covariates_table.md"
    table_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {table_path}")
    for fig_path in figure_paths:
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
