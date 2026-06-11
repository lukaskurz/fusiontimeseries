"""Analysis of the Phase-2 selection grid: table, figures, significance tests.

Loads ``results/few_shot_v2_selection/`` (plus ``results/few_shot_v2/`` for
the kNN-copy reference) and produces:

- a markdown results table (strategy x k per model, ID + OOD, random +-std)
  printed to stdout and written to ``docs/results/fewshot/selection_table.md``;
- the headline figure ``selection_random_vs_retrieval_vs_oracle.png``
  (4 models x 2 splits, RMSE vs k: random band, retrieval lines, oracle
  dashed, zero-shot and kNN-copy k=5 reference lines) plus a single-panel
  ``selection_headline_<best_model>.png``;
- paired comparisons (pairing="trace", bootstrap CI primary) at each
  strategy's best k: random vs each retrieval, op_knn vs each ctx_*,
  best retrieval vs oracle_tail (headroom), best retrieval vs kNN-copy k=5.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_selection
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
DEFAULT_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v2_selection"
DEFAULT_BASELINES_DIR: Path = REPO_ROOT / "results" / "few_shot_v2"
DEFAULT_OUT_DIR: Path = REPO_ROOT / "docs" / "results" / "fewshot"

RETRIEVAL_STRATEGIES: tuple[str, ...] = (
    "op_knn",
    "ctx_euclid",
    "ctx_dtw",
    "ctx_growth",
    "mmr_euclid",
)
TABLE_STRATEGIES: tuple[str, ...] = ("random",) + RETRIEVAL_STRATEGIES + ("oracle_tail",)
SPLITS: tuple[str, ...] = ("in_distribution", "out_of_distribution")
SPLIT_SHORT: dict[str, str] = {"in_distribution": "ID", "out_of_distribution": "OOD"}

STRATEGY_STYLE: dict[str, dict] = {
    "random": {"color": "0.35", "marker": "o", "label": "random (multi-seed)"},
    "op_knn": {"color": "tab:blue", "marker": "s", "label": "op_knn"},
    "ctx_euclid": {"color": "tab:orange", "marker": "^", "label": "ctx_euclid"},
    "ctx_dtw": {"color": "tab:green", "marker": "v", "label": "ctx_dtw"},
    "ctx_growth": {"color": "tab:purple", "marker": "D", "label": "ctx_growth"},
    "mmr_euclid": {"color": "tab:brown", "marker": "P", "label": "mmr_euclid"},
    "oracle_tail": {"color": "tab:red", "marker": "*", "label": "oracle_tail (cheating)"},
}

INTERPRETATION_NOTES = """
## Interpretation notes

- **oracle_tail is a cheating diagnostic**, not a method: it selects pool
  examples by the query's ground-truth tail mean. It is nearest-LABEL
  selection, not a model-in-the-loop upper bound — it shows how much signal
  perfectly level-matched examples carry under the current presentation
  format, not the best any selector could do.
- **Per-example z-scoring caveat**: the ICL pipeline z-scores every example
  independently, which erases absolute level — the very signal retrieval is
  supposed to inject (the metric is tail-LEVEL RMSE). If oracle_tail lands
  near random, the conclusion is "the presentation format hides level
  information", which hands off to Phase 3's shared-scaling ablation rather
  than condemning retrieval per se.
- TimesFM k=10 full-length example contexts exceed its max_context=2048 and
  are left-truncated — identical to the Phase-1 protocol, kept for
  comparability.
- MPS is not bit-deterministic: deterministic selection does not imply
  bit-identical RMSE across re-runs. All comparisons here live within one
  grid run.
- Wilcoxon p floors at 0.031 (n=6 ID) / 0.0625 (n=5 OOD) under
  pairing="trace"; the bootstrap CI is the primary evidence.
""".strip()


def build_index(results: list[dict]) -> dict[tuple[str, str, int], dict]:
    """Index results by (model_slug, strategy, k); latest timestamp wins.

    The strategy is recovered by stripping the ``{slug with / -> _}_`` prefix
    from the method label (the grid runner's naming convention).
    """
    index: dict[tuple[str, str, int], dict] = {}
    for r in results:
        slug = r.get("config", {}).get("model_slug")
        if slug is None or slug not in MODEL_SLUGS.values():
            continue
        prefix = slug.replace("/", "_") + "_"
        method = r.get("method", "")
        if not method.startswith(prefix):
            continue
        strategy = method[len(prefix) :]
        key = (slug, strategy, int(r["config"]["k_shot"]))
        if key not in index or r["timestamp"] > index[key]["timestamp"]:
            index[key] = r
    return index


def rmse_of(result: dict, split: str) -> float:
    return float(result[split]["rmse"])


def std_of(result: dict, split: str) -> float | None:
    value = result[split].get("rmse_std_seeds")
    return None if value is None else float(value)


def best_k(
    index: dict, slug: str, strategy: str, ks: list[int]
) -> tuple[int, dict] | None:
    """The (k, result) with minimal ID RMSE for a (model, strategy)."""
    candidates = [
        (k, index[(slug, strategy, k)]) for k in ks if (slug, strategy, k) in index
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda kr: rmse_of(kr[1], "in_distribution"))


def fmt_cell(result: dict | None) -> str:
    if result is None:
        return "—"
    parts = []
    for split in SPLITS:
        rmse = rmse_of(result, split)
        std = std_of(result, split)
        parts.append(f"{rmse:.2f}±{std:.2f}" if std is not None else f"{rmse:.2f}")
    return " / ".join(parts)


def fmt_comparison(label: str, comp: PairedComparison) -> str:
    wilcoxon = "n/a" if comp.wilcoxon_p is None else f"{comp.wilcoxon_p:.3f}"
    return (
        f"| {label} | {SPLIT_SHORT[comp.split]} | {comp.rmse_a:.2f} | {comp.rmse_b:.2f} | "
        f"{comp.rmse_diff:+.2f} | [{comp.bootstrap_ci_low:.2f}, {comp.bootstrap_ci_high:.2f}] | "
        f"{comp.bootstrap_p:.3f} | {wilcoxon} |"
    )


def make_table(index: dict, ks: list[int], knn_copy: dict | None) -> list[str]:
    lines = ["# Few-shot example selection — Phase 2 results", ""]
    lines.append(
        "Protocol: fixed 245-trace pool, full-length example targets (t266), "
        "context 80, prediction 64, tail 80. Cells are `ID / OOD` tail RMSE; "
        "`random` is mean±std over 20 selection seeds, deterministic "
        "strategies are a single pass (seed 42). Most-similar example is "
        "placed LAST (adjacent to the query)."
    )
    if knn_copy is not None:
        lines.append(
            f"\nModel-free reference — kNN-copy k=5: "
            f"**{rmse_of(knn_copy, 'in_distribution'):.2f} ID / "
            f"{rmse_of(knn_copy, 'out_of_distribution'):.2f} OOD**."
        )
    for slug in MODEL_SLUGS.values():
        if not any(key[0] == slug for key in index):
            continue
        lines.append(f"\n## {slug}\n")
        zeroshot = index.get((slug, "zeroshot", 0))
        if zeroshot is not None:
            lines.append(
                f"Zero-shot anchor (k=0): "
                f"{rmse_of(zeroshot, 'in_distribution'):.2f} ID / "
                f"{rmse_of(zeroshot, 'out_of_distribution'):.2f} OOD\n"
            )
        header = "| strategy | " + " | ".join(f"k={k}" for k in ks) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(ks) + 1))
        for strategy in TABLE_STRATEGIES:
            cells = [fmt_cell(index.get((slug, strategy, k))) for k in ks]
            if all(cell == "—" for cell in cells):
                continue
            lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    return lines


def plot_panel(
    ax: plt.Axes,
    index: dict,
    slug: str,
    split: str,
    ks: list[int],
    knn_copy: dict | None,
) -> None:
    for strategy in TABLE_STRATEGIES:
        points = [
            (k, index[(slug, strategy, k)]) for k in ks if (slug, strategy, k) in index
        ]
        if not points:
            continue
        xs = [k for k, _ in points]
        ys = [rmse_of(r, split) for _, r in points]
        style = STRATEGY_STYLE[strategy]
        linestyle = "--" if strategy == "oracle_tail" else "-"
        linewidth = 2.2 if strategy == "random" else 1.6
        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=5,
            label=style["label"],
        )
        if strategy == "random":
            stds = [std_of(r, split) or 0.0 for _, r in points]
            ax.fill_between(
                xs,
                [y - s for y, s in zip(ys, stds)],
                [y + s for y, s in zip(ys, stds)],
                color=style["color"],
                alpha=0.18,
                label="random ±1 std",
            )
    zeroshot = index.get((slug, "zeroshot", 0))
    if zeroshot is not None:
        ax.axhline(
            rmse_of(zeroshot, split),
            color="0.6",
            linestyle=":",
            linewidth=1.2,
            label="zero-shot (k=0)",
        )
    if knn_copy is not None:
        ax.axhline(
            rmse_of(knn_copy, split),
            color="black",
            linestyle="-.",
            linewidth=1.2,
            label="kNN-copy k=5",
        )
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)


def make_figures(
    index: dict, ks: list[int], knn_copy: dict | None, out_dir: Path
) -> tuple[Path, Path, str]:
    """Write the model-grid figure and the single-panel headline figure."""
    slugs = [slug for slug in MODEL_SLUGS.values() if any(key[0] == slug for key in index)]
    fig, axes = plt.subplots(
        len(slugs), 2, figsize=(11, 3.0 * len(slugs)), sharex=True, squeeze=False
    )
    for row, slug in enumerate(slugs):
        for col, split in enumerate(SPLITS):
            ax = axes[row][col]
            plot_panel(ax, index, slug, split, ks, knn_copy)
            ax.set_title(f"{slug.split('/')[-1]} — {SPLIT_SHORT[split]}", fontsize=10)
            if col == 0:
                ax.set_ylabel("tail RMSE")
            if row == len(slugs) - 1:
                ax.set_xlabel("k (examples)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("Few-shot example selection: random vs retrieval vs oracle", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    grid_path = out_dir / "selection_random_vs_retrieval_vs_oracle.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    # Headline: best model by ID RMSE over retrieval strategies, ID split only
    best_slug, best_rmse = None, float("inf")
    for (slug, strategy, _k), result in index.items():
        if strategy in RETRIEVAL_STRATEGIES:
            rmse = rmse_of(result, "in_distribution")
            if rmse < best_rmse:
                best_slug, best_rmse = slug, rmse
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_panel(ax, index, best_slug, "in_distribution", ks, knn_copy)
    ax.set_xlabel("k (examples)")
    ax.set_ylabel("tail RMSE (in-distribution)")
    ax.set_title(f"Example selection — {best_slug} (ID)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    model_short = best_slug.split("/")[-1].lower().replace(".", "p")
    headline_path = out_dir / f"selection_headline_{model_short}.png"
    fig.savefig(headline_path, dpi=150)
    plt.close(fig)
    return grid_path, headline_path, best_slug


def make_comparisons(index: dict, ks: list[int], knn_copy: dict | None) -> list[str]:
    """Paired comparisons (pairing='trace') at each strategy's best-ID k."""
    lines = [
        "\n## Paired comparisons (pairing=trace, bootstrap CI primary)",
        "",
        "Each strategy enters at its own best k (chosen by ID RMSE). "
        "Δ = RMSE(A) − RMSE(B); negative favours A. CI/p from a 10k paired "
        "bootstrap over traces; Wilcoxon shown for completeness (p floors "
        "0.031 ID / 0.0625 OOD).",
    ]
    header = (
        "\n| A vs B | split | RMSE A | RMSE B | Δ(A−B) | 95% CI | p_boot | p_wilcoxon |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    for slug in MODEL_SLUGS.values():
        if not any(key[0] == slug for key in index):
            continue
        bests = {
            strategy: best_k(index, slug, strategy, ks)
            for strategy in TABLE_STRATEGIES
        }
        retrieval_bests = {
            s: kr for s, kr in bests.items() if s in RETRIEVAL_STRATEGIES and kr
        }
        if not retrieval_bests:
            continue
        best_strategy = min(
            retrieval_bests, key=lambda s: rmse_of(retrieval_bests[s][1], "in_distribution")
        )
        lines.append(f"\n### {slug}")
        lines.append(
            "Best ks: "
            + ", ".join(f"{s} k={kr[0]}" for s, kr in bests.items() if kr is not None)
            + f"; best retrieval: **{best_strategy}**."
        )
        lines.append(header)

        pairs: list[tuple[str, dict, str, dict]] = []
        if bests.get("random"):
            k_r, random_result = bests["random"]
            for strategy, (k_s, result) in retrieval_bests.items():
                pairs.append(
                    (f"random(k={k_r})", random_result, f"{strategy}(k={k_s})", result)
                )
        if bests.get("op_knn"):
            k_o, op_result = bests["op_knn"]
            for strategy in ("ctx_euclid", "ctx_dtw", "ctx_growth"):
                if bests.get(strategy):
                    k_s, result = bests[strategy]
                    pairs.append(
                        (f"op_knn(k={k_o})", op_result, f"{strategy}(k={k_s})", result)
                    )
        k_b, best_result = retrieval_bests[best_strategy]
        if bests.get("oracle_tail"):
            k_or, oracle_result = bests["oracle_tail"]
            pairs.append(
                (
                    f"{best_strategy}(k={k_b})",
                    best_result,
                    f"oracle_tail(k={k_or})",
                    oracle_result,
                )
            )
        if knn_copy is not None:
            pairs.append(
                (f"{best_strategy}(k={k_b})", best_result, "kNN-copy(k=5)", knn_copy)
            )

        for label_a, result_a, label_b, result_b in pairs:
            for split in SPLITS:
                comp = paired_comparison(result_a, result_b, split, pairing="trace")
                lines.append(fmt_comparison(f"{label_a} vs {label_b}", comp))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Phase-2 selection grid")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    index = build_index(load_results(args.results_dir))
    assert index, f"No selection-grid results in {args.results_dir}"
    ks = sorted({k for (_, strategy, k) in index if strategy != "zeroshot"})

    knn_copy = None
    for r in load_results(args.baselines_dir):
        if r.get("method") == "knn_copy_k5":
            knn_copy = r
    if knn_copy is None:
        print(f"WARNING: no knn_copy_k5 result in {args.baselines_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = make_table(index, ks, knn_copy)
    grid_path, headline_path, best_slug = make_figures(index, ks, knn_copy, args.out_dir)
    lines.append(
        f"\n![selection grid]({grid_path.name})\n\n![headline]({headline_path.name})"
    )
    lines.extend(make_comparisons(index, ks, knn_copy))
    lines.append("\n" + INTERPRETATION_NOTES)

    markdown = "\n".join(lines) + "\n"
    table_path = args.out_dir / "selection_table.md"
    table_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {table_path}")
    print(f"Wrote {grid_path}")
    print(f"Wrote {headline_path} (best model: {best_slug})")


if __name__ == "__main__":
    main()
