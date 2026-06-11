"""Analysis of the Phase-3 presentation grid: table, figures, significance.

Loads ``results/few_shot_v3_presentation/`` (plus ``results/few_shot_v2/``
for the kNN-copy reference and ``results/few_shot_v2_selection/`` as
clearly-marked cross-run context) and produces, in ``docs/results/fewshot/``:

- ``presentation_table.md`` — all four ablations (A group vs concat,
  B normalization, C ordering, D truncation) + paired comparisons +
  interpretation notes;
- ``presentation_group_vs_concat.png`` — Chronos-2 k-curve 0..20, solid
  group vs faded concat, random band, oracle dashed, zero-shot + kNN-copy
  references, context-clamp annotation;
- ``presentation_norm_ablation.png`` — per model: random / ctx_euclid /
  oracle_tail under per-example (faded) vs shared (solid) scaling.

Method labels are parsed as ``{slug with / -> _}_{strategy}__{variant}``
(variant tokens: base | group; shared; simfirst | shuforder; truncN) and
cross-checked against the FewShotConfig metadata fields. The analyzer
hard-asserts example-id equality for every ``__group`` / ``__base`` pair.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_presentation
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusiontimeseries.benchmarking.few_shot.harness import (
    PairedComparison,
    load_results,
    paired_comparison,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import MODEL_SLUGS

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v3_presentation"
DEFAULT_BASELINES_DIR: Path = REPO_ROOT / "results" / "few_shot_v2"
DEFAULT_SELECTION_DIR: Path = REPO_ROOT / "results" / "few_shot_v2_selection"
DEFAULT_OUT_DIR: Path = REPO_ROOT / "docs" / "results" / "fewshot"

CHRONOS2_SLUG = MODEL_SLUGS["chronos2"]
SPLITS: tuple[str, ...] = ("in_distribution", "out_of_distribution")
SPLIT_SHORT: dict[str, str] = {"in_distribution": "ID", "out_of_distribution": "OOD"}
GROUP_STRATEGIES: tuple[str, ...] = ("random", "ctx_euclid", "oracle_tail")
SHARED_STRATEGIES: tuple[str, ...] = (
    "random",
    "op_knn",
    "ctx_euclid",
    "ctx_dtw",
    "ctx_growth",
    "mmr_euclid",
    "oracle_tail",
)

#: Index key: (slug, strategy, k, presentation, normalization, order, trunc)
Key = tuple[str, str, int, str, str, str, int | None]

STRATEGY_STYLE: dict[str, dict] = {
    "random": {"color": "0.35", "marker": "o"},
    "op_knn": {"color": "tab:blue", "marker": "s"},
    "ctx_euclid": {"color": "tab:orange", "marker": "^"},
    "ctx_dtw": {"color": "tab:green", "marker": "v"},
    "ctx_growth": {"color": "tab:purple", "marker": "D"},
    "mmr_euclid": {"color": "tab:brown", "marker": "P"},
    "oracle_tail": {"color": "tab:red", "marker": "*"},
}

INTERPRETATION_NOTES = """
## Interpretation notes

- **Group ICL and shared scaling are complementary fixes.** Chronos-2
  instance-norms each row of a group task independently, so our outer
  scalers are mathematically inert for it — group ICL can only fix the
  splice-discontinuity artifact of flat concatenation, never restore
  absolute level. Shared scaling restores level but keeps the splices.
- **Shared scaling** fits ONE StandardScaler on the query's 80-step context
  and applies it to example contexts+targets and the query: an example's
  absolute level survives the normalize → predict → denormalize round trip
  (per-example z-scoring erases it — the Phase-2 diagnosis). k=0 reduces
  exactly to zero-shot. Model-free analogue: kNN-copy with rescale=False
  (absolute level copy) vs rescale=True (amplitude transfer ≈ per-example).
- **oracle_tail is a cheating diagnostic**, not a method: it selects pool
  examples by the query's ground-truth tail mean. oracle__shared vs
  random__shared is the headroom of label-aware selection once the
  presentation can carry level; on OOD the pool may simply not contain
  tails at OOD levels — that limitation is part of the result.
- **Context budgets, not just presentation**: concat at k=20 is ~5420 steps;
  Chronos-2 left-clamps per-row context to 2048 (≈ the last 7 examples plus
  the query survive) and TimesFM's max_context is 2048 (full k=10 ≈ 2750 is
  already left-truncated, identical to the Phase-1/2 protocol). Truncated
  examples (peak+64, mean length ≈ 130) make k=10 fit (~1383) while k=20
  (~2686) truncates again — so trunc k=20 still partially measures the
  window, not only the example count.
- MPS is not bit-deterministic across process runs: headline comparisons
  live within the single v3 grid run; v2_selection numbers are cross-run
  context only.
- Wilcoxon p floors at 0.031 (n=6 ID) / 0.0625 (n=5 OOD) under
  pairing="trace"; the bootstrap CI is the primary evidence. Where seed
  sets are identical, a pairing="trace_seed" row adds resolution but treats
  seeds as independent.
""".strip()


########################################################
# Parsing and indexing
########################################################


def parse_variant(variant: str) -> dict | None:
    """Decode a hyphen-joined variant label into presentation fields."""
    fields = {
        "presentation": "concat",
        "normalization": "per_example",
        "example_order": "similar_last",
        "example_truncation_margin": None,
    }
    if variant == "base":
        return fields
    for token in variant.split("-"):
        if token == "group":
            fields["presentation"] = "group"
        elif token == "shared":
            fields["normalization"] = "shared"
        elif token == "simfirst":
            fields["example_order"] = "similar_first"
        elif token == "shuforder":
            fields["example_order"] = "shuffled"
        elif token.startswith("trunc") and token[5:].isdigit():
            fields["example_truncation_margin"] = int(token[5:])
        else:
            return None
    return fields


def build_index(results: list[dict]) -> dict[Key, dict]:
    """Index v3 results by the full presentation key; latest timestamp wins.

    Methods without a ``__variant`` suffix are skipped (not Phase-3 files).
    Variant tokens are cross-checked against the FewShotConfig metadata
    fields the grid runner wrote.
    """
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
            fields["normalization"],
            fields["example_order"],
            fields["example_truncation_margin"],
        )
        if key not in index or r["timestamp"] > index[key]["timestamp"]:
            index[key] = r
    return index


def get(
    index: dict[Key, dict],
    slug: str,
    strategy: str,
    k: int,
    presentation: str = "concat",
    norm: str = "per_example",
    order: str = "similar_last",
    trunc: int | None = None,
) -> dict | None:
    return index.get((slug, strategy, k, presentation, norm, order, trunc))


def rmse_of(result: dict, split: str) -> float:
    return float(result[split]["rmse"])


def std_of(result: dict, split: str) -> float | None:
    value = result[split].get("rmse_std_seeds")
    return None if value is None else float(value)


def best_k(
    index: dict[Key, dict], slug: str, strategy: str, ks: list[int], **variant
) -> tuple[int, dict] | None:
    """The (k, result) with minimal ID RMSE for a (model, strategy, variant)."""
    candidates = [
        (k, result)
        for k in ks
        if (result := get(index, slug, strategy, k, **variant)) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda kr: rmse_of(kr[1], "in_distribution"))


def assert_group_base_example_ids(index: dict[Key, dict]) -> int:
    """Hard-assert per-seed example-id equality for __group / __base pairs."""
    checked = 0
    for key, group_result in index.items():
        slug, strategy, k, presentation, norm, order, trunc = key
        if presentation != "group" or strategy == "zeroshot":
            continue
        base = index.get((slug, strategy, k, "concat", norm, order, trunc))
        assert base is not None, f"Missing __base twin for group cell {key}"
        assert group_result["seeds"] == base["seeds"], f"Seed mismatch for {key}"
        for sg, sb in zip(group_result["per_seed"], base["per_seed"]):
            assert sg["example_ids"] == sb["example_ids"], (
                f"example_ids differ for {key} seed {sg['seed']}"
            )
        checked += 1
    return checked


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
    index: dict[Key, dict],
    slug: str,
    rows: list[tuple[str, str, dict]],
    ks: list[int],
) -> list[str]:
    """One markdown table: rows = (label, strategy, variant kwargs)."""
    lines = ["| variant | " + " | ".join(f"k={k}" for k in ks) + " |"]
    lines.append("|" + "---|" * (len(ks) + 1))
    for label, strategy, variant in rows:
        cells = [fmt_cell(get(index, slug, strategy, k, **variant)) for k in ks]
        if all(cell == "—" for cell in cells):
            continue
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def zeroshot_line(index: dict[Key, dict], slug: str) -> str | None:
    anchor = get(index, slug, "zeroshot", 0)
    if anchor is None:
        return None
    return (
        f"Zero-shot anchor (k=0): {rmse_of(anchor, 'in_distribution'):.2f} ID / "
        f"{rmse_of(anchor, 'out_of_distribution'):.2f} OOD"
    )


def make_table(
    index: dict[Key, dict], knn_copy: dict | None
) -> list[str]:
    lines = ["# Few-shot example presentation — Phase 3 results", ""]
    lines.append(
        "Protocol: fixed 245-trace pool, full-length example targets (t266), "
        "context 80, prediction 64, tail 80; same harness and metric as "
        "Phase 2. Cells are `ID / OOD` tail RMSE; multi-seed cells are "
        "mean±std over selection seeds (random: 20, shuffled order: 5), "
        "deterministic cells a single pass (seed 42). All cells come from "
        "ONE grid run (`results/few_shot_v3_presentation/`)."
    )
    if knn_copy is not None:
        lines.append(
            f"\nModel-free reference — kNN-copy k=5 (absolute level copy, "
            f"rescale=False): **{rmse_of(knn_copy, 'in_distribution'):.2f} ID / "
            f"{rmse_of(knn_copy, 'out_of_distribution'):.2f} OOD**."
        )

    # A — group vs concat (chronos2)
    a_ks = sorted(
        {key[2] for key in index if key[0] == CHRONOS2_SLUG and key[3] == "group" and key[2] > 0}
    )
    if a_ks:
        lines.append("\n## A — Chronos-2: group ICL vs flat concat\n")
        lines.append(
            "Examples as group rows (`past_covariates` of one dict task, "
            "GroupSelfAttention) vs the Phase-1/2 spliced concatenation — "
            "identical example sets per cell pair (hard-asserted). Note "
            "concat at k=20 is ~5420 steps, left-clamped to Chronos-2's "
            "2048-step context (≈ last 7 examples visible).\n"
        )
        anchor = zeroshot_line(index, CHRONOS2_SLUG)
        if anchor:
            lines.append(anchor + " (identical for both presentations)\n")
        rows = []
        for strategy in GROUP_STRATEGIES:
            rows.append((f"{strategy} (concat)", strategy, {"presentation": "concat"}))
            rows.append((f"{strategy} (group)", strategy, {"presentation": "group"}))
        lines.extend(table_block(index, CHRONOS2_SLUG, rows, a_ks))

    # B — normalization
    b_slugs = [
        slug
        for slug in MODEL_SLUGS.values()
        if any(key[0] == slug and key[4] == "shared" and key[6] is None for key in index)
    ]
    if b_slugs:
        lines.append("\n## B — Normalization: per-example vs shared scaling\n")
        lines.append(
            "`shared` = ONE scaler fit on the query context applied to "
            "examples and query (absolute level survives); `per-example` = "
            "the Phase-1/2 default (level erased). Full strategy grid under "
            "shared; per-example anchors for random / ctx_euclid / "
            "oracle_tail.\n"
        )
        for slug in b_slugs:
            b_ks = sorted(
                {
                    key[2]
                    for key in index
                    if key[0] == slug and key[4] == "shared" and key[6] is None
                }
            )
            lines.append(f"\n### {slug}\n")
            anchor = zeroshot_line(index, slug)
            if anchor:
                lines.append(anchor + "\n")
            rows = []
            for strategy in SHARED_STRATEGIES:
                rows.append(
                    (f"{strategy} (per-example)", strategy, {"norm": "per_example"})
                )
                rows.append((f"{strategy} (shared)", strategy, {"norm": "shared"}))
            lines.extend(table_block(index, slug, rows, b_ks))

    # C — ordering
    c_slugs = [
        slug
        for slug in MODEL_SLUGS.values()
        if any(key[0] == slug and key[5] != "similar_last" for key in index)
    ]
    if c_slugs:
        lines.append("\n## C — Ordering (ctx_euclid, per-example norm)\n")
        lines.append(
            "Phase-2 convention is most-similar LAST (adjacent to the "
            "query). `shuffled` permutes deterministically per (seed, "
            "query); k=1 is order-free.\n"
        )
        for slug in c_slugs:
            c_ks = sorted(
                {key[2] for key in index if key[0] == slug and key[5] != "similar_last"}
            )
            lines.append(f"\n### {slug}\n")
            rows = [
                ("similar_last (anchor)", "ctx_euclid", {}),
                ("similar_first", "ctx_euclid", {"order": "similar_first"}),
                ("shuffled", "ctx_euclid", {"order": "shuffled"}),
            ]
            lines.extend(table_block(index, slug, rows, c_ks))

    # D — truncation
    d_cells = [key for key in index if key[6] is not None]
    if d_cells:
        trunc_norm = d_cells[0][4]
        margin = d_cells[0][6]
        d_slugs = sorted({key[0] for key in d_cells}, key=list(MODEL_SLUGS.values()).index)
        lines.append(f"\n## D — Truncated examples (peak+{margin}, {trunc_norm} norm)\n")
        lines.append(
            "Truncation applied AFTER selection (rankings see full traces). "
            "Mean truncated length ≈ 130 vs full 267: k=10 fits TimesFM's "
            "2048-step window (~1383 steps), k=20 (~2686) truncates again — "
            "k=20 cells still partially measure the window.\n"
        )
        for slug in d_slugs:
            d_ks = sorted({key[2] for key in index if key[0] == slug and key[6] is not None})
            lines.append(f"\n### {slug}\n")
            rows = []
            for strategy in ("random", "ctx_euclid"):
                rows.append(
                    (f"{strategy} (full)", strategy, {"norm": trunc_norm})
                )
                rows.append(
                    (
                        f"{strategy} (trunc{margin})",
                        strategy,
                        {"norm": trunc_norm, "trunc": margin},
                    )
                )
            lines.extend(table_block(index, slug, rows, d_ks))

    return lines


########################################################
# Figures
########################################################


def _reference_lines(ax: plt.Axes, index: dict[Key, dict], slug: str, split: str, knn_copy: dict | None) -> None:
    anchor = get(index, slug, "zeroshot", 0)
    if anchor is not None:
        ax.axhline(
            rmse_of(anchor, split), color="0.6", linestyle=":", linewidth=1.2,
            label="zero-shot (k=0)",
        )
    if knn_copy is not None:
        ax.axhline(
            rmse_of(knn_copy, split), color="black", linestyle="-.", linewidth=1.2,
            label="kNN-copy k=5",
        )


def fig_group_vs_concat(
    index: dict[Key, dict], knn_copy: dict | None, out_dir: Path
) -> Path | None:
    slug = CHRONOS2_SLUG
    ks = sorted({key[2] for key in index if key[0] == slug and key[3] == "group" and key[2] > 0})
    if not ks:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    for ax, split in zip(axes, SPLITS):
        for strategy in GROUP_STRATEGIES:
            style = STRATEGY_STYLE[strategy]
            linestyle = "--" if strategy == "oracle_tail" else "-"
            for presentation, alpha, lw in (("concat", 0.35, 1.4), ("group", 1.0, 2.0)):
                points = [
                    (k, r)
                    for k in ks
                    if (r := get(index, slug, strategy, k, presentation=presentation))
                ]
                if not points:
                    continue
                xs = [k for k, _ in points]
                ys = [rmse_of(r, split) for _, r in points]
                ax.plot(
                    xs, ys, color=style["color"], marker=style["marker"],
                    linestyle=linestyle, linewidth=lw, alpha=alpha, markersize=5,
                    label=f"{strategy} ({presentation})",
                )
                if strategy == "random":
                    stds = [std_of(r, split) or 0.0 for _, r in points]
                    ax.fill_between(
                        xs,
                        [y - s for y, s in zip(ys, stds)],
                        [y + s for y, s in zip(ys, stds)],
                        color=style["color"], alpha=0.12 * alpha + 0.06,
                    )
        _reference_lines(ax, index, slug, split, knn_copy)
        ax.set_xticks([0] + ks)
        ax.set_xlabel("k (examples)")
        ax.grid(alpha=0.3)
        ax.set_title(f"Chronos-2 — {SPLIT_SHORT[split]}", fontsize=11)
    axes[0].set_ylabel("tail RMSE")
    if 20 in ks:
        axes[0].annotate(
            "concat k=20 ≈ 5420 steps,\nleft-clamped to 2048\n(~last 7 examples visible)",
            xy=(20, axes[0].get_ylim()[0]),
            xytext=(0.62, 0.04), textcoords="axes fraction",
            fontsize=7.5, color="0.3",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Chronos-2 group ICL (solid) vs flat concat (faded), identical example sets", y=0.98)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    path = out_dir / "presentation_group_vs_concat.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_norm_ablation(
    index: dict[Key, dict], knn_copy: dict | None, out_dir: Path
) -> Path | None:
    slugs = [
        slug
        for slug in MODEL_SLUGS.values()
        if any(key[0] == slug and key[4] == "shared" and key[6] is None for key in index)
    ]
    if not slugs:
        return None
    fig, axes = plt.subplots(
        len(slugs), 2, figsize=(11, 3.0 * len(slugs)), sharex=True, squeeze=False
    )
    for row, slug in enumerate(slugs):
        ks = sorted(
            {key[2] for key in index if key[0] == slug and key[4] == "shared" and key[6] is None}
        )
        for col, split in enumerate(SPLITS):
            ax = axes[row][col]
            for strategy in GROUP_STRATEGIES:
                style = STRATEGY_STYLE[strategy]
                for norm, alpha, lw, ls in (
                    ("per_example", 0.35, 1.4, "--"),
                    ("shared", 1.0, 2.0, "-"),
                ):
                    points = [
                        (k, r)
                        for k in ks
                        if (r := get(index, slug, strategy, k, norm=norm))
                    ]
                    if not points:
                        continue
                    xs = [k for k, _ in points]
                    ys = [rmse_of(r, split) for _, r in points]
                    ax.plot(
                        xs, ys, color=style["color"], marker=style["marker"],
                        linestyle=ls, linewidth=lw, alpha=alpha, markersize=5,
                        label=f"{strategy} ({'shared' if norm == 'shared' else 'per-ex'})",
                    )
                    if strategy == "random":
                        stds = [std_of(r, split) or 0.0 for _, r in points]
                        ax.fill_between(
                            xs,
                            [y - s for y, s in zip(ys, stds)],
                            [y + s for y, s in zip(ys, stds)],
                            color=style["color"], alpha=0.12 * alpha + 0.06,
                        )
            _reference_lines(ax, index, slug, split, knn_copy)
            ax.set_xticks(ks)
            ax.grid(alpha=0.3)
            ax.set_title(f"{slug.split('/')[-1]} — {SPLIT_SHORT[split]}", fontsize=10)
            if col == 0:
                ax.set_ylabel("tail RMSE")
            if row == len(slugs) - 1:
                ax.set_xlabel("k (examples)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Normalization ablation: shared scaling (solid) vs per-example (faded)", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    path = out_dir / "presentation_norm_ablation.png"
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
    """Both splits, pairing='trace'; plus 'trace_seed' when seeds match."""
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


def make_comparisons(index: dict[Key, dict], knn_copy: dict | None) -> list[str]:
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

    # A — group vs base per strategy: at base-best k and at max k
    slug = CHRONOS2_SLUG
    a_ks = sorted({key[2] for key in index if key[0] == slug and key[3] == "group" and key[2] > 0})
    if a_ks:
        lines.append(f"\n### A — group vs concat ({slug})")
        lines.append(header)
        for strategy in GROUP_STRATEGIES:
            base_best = best_k(index, slug, strategy, a_ks, presentation="concat")
            if base_best is None:
                continue
            compare_ks = {base_best[0], max(a_ks)}
            for k in sorted(compare_ks):
                g = get(index, slug, strategy, k, presentation="group")
                b = get(index, slug, strategy, k, presentation="concat")
                if g is None or b is None:
                    continue
                lines.extend(comparison_rows(f"{strategy} group(k={k}) vs concat(k={k})", g, b))

    # B — normalization comparisons per model
    for slug in MODEL_SLUGS.values():
        b_ks = sorted(
            {key[2] for key in index if key[0] == slug and key[4] == "shared" and key[6] is None}
        )
        if not b_ks:
            continue
        lines.append(f"\n### B — normalization ({slug})")
        lines.append(header)
        bests = {
            (strategy, norm): best_k(index, slug, strategy, b_ks, norm=norm)
            for strategy in ("random", "ctx_euclid", "oracle_tail")
            for norm in ("per_example", "shared")
        }

        def add_pair(label_a: str, kr_a, label_b: str, kr_b) -> None:
            if kr_a is None or kr_b is None:
                return
            (k_a, res_a), (k_b, res_b) = kr_a, kr_b
            lines.extend(
                comparison_rows(f"{label_a}(k={k_a}) vs {label_b}(k={k_b})", res_a, res_b)
            )

        # The critical cell: does the oracle finally beat its own base?
        add_pair(
            "oracle__shared", bests[("oracle_tail", "shared")],
            "oracle__base", bests[("oracle_tail", "per_example")],
        )
        # THE result: headroom under a level-carrying presentation
        add_pair(
            "oracle__shared", bests[("oracle_tail", "shared")],
            "random__shared", bests[("random", "shared")],
        )
        # Does retrieval finally pay once level survives?
        add_pair(
            "ctx_euclid__shared", bests[("ctx_euclid", "shared")],
            "random__shared", bests[("random", "shared")],
        )
        add_pair(
            "random__shared", bests[("random", "shared")],
            "random__base", bests[("random", "per_example")],
        )

    # C — ordering comparisons (same k, base-best among the C ks)
    for slug in MODEL_SLUGS.values():
        c_ks = sorted({key[2] for key in index if key[0] == slug and key[5] != "similar_last"})
        if not c_ks:
            continue
        anchor_best = best_k(index, slug, "ctx_euclid", c_ks)
        if anchor_best is None:
            continue
        k = anchor_best[0]
        anchor = anchor_best[1]
        lines.append(f"\n### C — ordering ({slug}, ctx_euclid, k={k})")
        lines.append(header)
        for order, label in (("similar_first", "simfirst"), ("shuffled", "shuforder")):
            other = get(index, slug, "ctx_euclid", k, order=order)
            if other is not None:
                lines.extend(comparison_rows(f"{label}(k={k}) vs similar_last(k={k})", other, anchor))

    # D — truncation comparisons
    d_cells = sorted({(key[0], key[4], key[6]) for key in index if key[6] is not None})
    for slug, trunc_norm, margin in d_cells:
        d_ks = sorted(
            {key[2] for key in index if key[0] == slug and key[6] == margin}
        )
        lines.append(f"\n### D — truncation ({slug}, {trunc_norm} norm)")
        lines.append(header)
        for strategy in ("random", "ctx_euclid"):
            for k in d_ks:
                tr = get(index, slug, strategy, k, norm=trunc_norm, trunc=margin)
                full = get(index, slug, strategy, k, norm=trunc_norm)
                if tr is not None and full is not None:
                    lines.extend(
                        comparison_rows(f"{strategy} trunc{margin}(k={k}) vs full(k={k})", tr, full)
                    )
            # Budget argument: more (truncated) examples vs fewer full ones
            tr20 = get(index, slug, strategy, 20, norm=trunc_norm, trunc=margin)
            full10 = get(index, slug, strategy, 10, norm=trunc_norm)
            if tr20 is not None and full10 is not None:
                lines.extend(
                    comparison_rows(f"{strategy} trunc{margin}(k=20) vs full(k=10)", tr20, full10)
                )

    # Best legitimate v3 cells vs kNN-copy
    if knn_copy is not None:
        lines.append("\n### Best v3 cells (non-oracle) vs kNN-copy k=5")
        lines.append(header)
        for slug in MODEL_SLUGS.values():
            candidates = [
                (key, r)
                for key, r in index.items()
                if key[0] == slug and key[1] not in ("oracle_tail", "zeroshot")
            ]
            if not candidates:
                continue
            best_key, best_result = min(
                candidates, key=lambda kr: rmse_of(kr[1], "in_distribution")
            )
            _, strategy, k, presentation, norm, order, trunc = best_key
            tokens = [strategy, f"k={k}"]
            if presentation != "concat":
                tokens.append(presentation)
            if norm != "per_example":
                tokens.append(norm)
            if order != "similar_last":
                tokens.append(order)
            if trunc is not None:
                tokens.append(f"trunc{trunc}")
            label = f"{slug.split('/')[-1]} {' '.join(tokens)}"
            lines.extend(comparison_rows(f"{label} vs kNN-copy(k=5)", best_result, knn_copy))

    return lines


########################################################
# Cross-run context (Phase 2)
########################################################


def make_cross_run_context(selection_dir: Path) -> list[str]:
    """Phase-2 best cells, clearly marked as a DIFFERENT grid run."""
    if not selection_dir.exists():
        return []
    lines = [
        "\n## Cross-run context — Phase 2 selection grid "
        "(different run; MPS drift possible, do not compare cell-to-cell)",
        "",
        "| model | random k=5 | ctx_euclid k=5 | oracle_tail k=5 |",
        "|---|---|---|---|",
    ]
    results = load_results(selection_dir)
    by_method_k: dict[tuple[str, int], dict] = {}
    for r in results:
        key = (r.get("method", ""), int(r.get("config", {}).get("k_shot", -1)))
        if key not in by_method_k or r["timestamp"] > by_method_k[key]["timestamp"]:
            by_method_k[key] = r
    for slug in MODEL_SLUGS.values():
        clean = slug.replace("/", "_")
        cells = []
        for strategy in ("random", "ctx_euclid", "oracle_tail"):
            r = by_method_k.get((f"{clean}_{strategy}", 5))
            cells.append(fmt_cell(r))
        if all(cell == "—" for cell in cells):
            continue
        lines.append(f"| {slug} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Phase-3 presentation grid")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    index = build_index(load_results(args.results_dir))
    assert index, f"No presentation-grid results in {args.results_dir}"
    n_pairs = assert_group_base_example_ids(index)
    print(f"✓ example_ids identical for all {n_pairs} __group/__base pairs")

    knn_copy = None
    for r in load_results(args.baselines_dir):
        if r.get("method") == "knn_copy_k5":
            if knn_copy is None or r["timestamp"] > knn_copy["timestamp"]:
                knn_copy = r
    if knn_copy is None:
        print(f"WARNING: no knn_copy_k5 result in {args.baselines_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = make_table(index, knn_copy)
    figure_paths = []
    for fig_path in (
        fig_group_vs_concat(index, knn_copy, args.out_dir),
        fig_norm_ablation(index, knn_copy, args.out_dir),
    ):
        if fig_path is not None:
            figure_paths.append(fig_path)
            lines.append(f"\n![{fig_path.stem}]({fig_path.name})")
    lines.extend(make_comparisons(index, knn_copy))
    lines.extend(make_cross_run_context(args.selection_dir))
    lines.append("\n" + INTERPRETATION_NOTES)

    markdown = "\n".join(lines) + "\n"
    table_path = args.out_dir / "presentation_table.md"
    table_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {table_path}")
    for fig_path in figure_paths:
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
