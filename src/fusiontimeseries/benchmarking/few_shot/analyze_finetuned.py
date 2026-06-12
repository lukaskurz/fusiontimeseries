"""Analysis of the Phase-6 finetuned-ICL grid: does adaptation stack?

Loads ``results/few_shot_v6_finetuned/`` (+ ``severin_anchor.json``) and
produces, in ``docs/results/fewshot/``:

- ``finetuned_icl_table.md`` — the 2x2 {base, finetuned} x {k=0, best-k}
  headline (mean decoding; median appendix), the full per-config table,
  paired comparisons (ft+ICL vs ft k0 / vs base+ICL; ft k0 vs base k0;
  base+ICL vs base k0), the window block (full 8192 vs the 512 training
  window), the Severin-protocol anchor pair (his ``mean(x[:-80])`` metric vs
  the honest ``mean(x[-80:])`` on the same forecasts), the v5 per-trace
  bridge report, and the synergy verdict;
- ``finetuned_synergy.png`` — per config, paired base→ft dots/arrows, two
  panels ID/OOD (mean decoding);
- ``adaptation_ladder.png`` — REGENERATED ladder whose finetuned rungs are
  measured v6 cells (our checkpoint, our harness, the honest tail metric);
  Severin's README finetuning rows are kept as annotated references — they
  were scored with ``mean(x[:-80])`` (head mean INCLUDING the 80 copied
  ground-truth context steps) and are NOT comparable to the tail-RMSE rungs.

Method labels parse as ``{slug}_{strategy}__{variant}`` with the v6 token
set (``shared``; ``mean``; ``win512``), cross-checked against the
FewShotConfig metadata (incl. ``checkpoint`` and ``model_context_window``).
Hard asserts: ONE checkpoint id across all ft cells; per-seed example-id
equality across decoding twins, base/ft twins, and window twins.

Caveat noted in captions: base cells are bf16 (protocol continuity with
v3-v5), finetuned cells fp32 (training numerics).

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_finetuned
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fusiontimeseries.benchmarking.few_shot.finetuned import (
    FINETUNED_SLUG,
    FT_TRAIN_CONTEXT,
)
from fusiontimeseries.benchmarking.few_shot.harness import (
    PairedComparison,
    load_results,
    paired_comparison,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import MODEL_SLUGS
from fusiontimeseries.benchmarking.few_shot.run_finetuned_grid import (
    ICL_CONFIGS,
    RANDOM_K,
    WINDOW_CONFIGS,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v6_finetuned"
V5_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v5_decoding"
DEFAULT_OUT_DIR: Path = REPO_ROOT / "docs" / "results" / "fewshot"

BASE_SLUG: str = MODEL_SLUGS["chronos2"]
SPLITS: tuple[str, ...] = ("in_distribution", "out_of_distribution")
SPLIT_SHORT: dict[str, str] = {"in_distribution": "ID", "out_of_distribution": "OOD"}
POINT_STATS: tuple[str, ...] = ("mean", "median")

#: Index key: (slug, strategy, k, point_stat, window) — window None = full.
Key = tuple[str, str, int, str, int | None]

#: All grid configs as (strategy, k) in display order.
GRID_CONFIGS: tuple[tuple[str, int], ...] = (
    ("zeroshot", 0),
    *ICL_CONFIGS,
    ("random", RANDOM_K),
)
LEGIT_ICL: tuple[tuple[str, int], ...] = tuple(
    (s, k) for s, k in ICL_CONFIGS if s != "oracle_tail"
)

#: v5-overlapping base configs (the cross-dir bridge).
BRIDGE_CONFIGS: tuple[tuple[str, int], ...] = (
    ("zeroshot", 0),
    ("mmr_euclid", 5),
    ("oracle_tail", 10),
    ("random", RANDOM_K),
)


########################################################
# Parsing and indexing
########################################################


def parse_variant(variant: str) -> dict | None:
    """Decode a v6 hyphen-joined variant label into config fields."""
    fields: dict = {
        "normalization": "per_example",
        "point_stat": "median",
        "model_context_window": None,
    }
    if variant == "base":
        return fields
    for token in variant.split("-"):
        if token == "shared":
            fields["normalization"] = "shared"
        elif token in ("mean", "meanhead"):
            fields["point_stat"] = token
        elif token.startswith("win") and token[3:].isdigit():
            fields["model_context_window"] = int(token[3:])
        else:
            return None
    return fields


def build_index(results: list[dict]) -> tuple[dict[Key, dict], str]:
    """Index v6 results; latest timestamp wins; tokens cross-checked.

    Returns:
        (index, checkpoint_id) — every finetuned cell must carry the SAME
        checkpoint id (hard assert).
    """
    index: dict[Key, dict] = {}
    checkpoints: set[str] = set()
    for r in results:
        config = r.get("config", {})
        slug = config.get("model_slug")
        if slug not in (BASE_SLUG, FINETUNED_SLUG):
            continue
        prefix = slug.replace("/", "_") + "_"
        method = r.get("method", "")
        if not method.startswith(prefix) or "__" not in method:
            continue
        strategy, variant = method[len(prefix):].rsplit("__", 1)
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
        if slug == FINETUNED_SLUG:
            assert config.get("checkpoint"), f"{method}: ft cell without checkpoint id"
            checkpoints.add(config["checkpoint"])
        else:
            assert config.get("checkpoint") is None, f"{method}: base cell w/ checkpoint"
        key: Key = (
            slug,
            strategy,
            int(config["k_shot"]),
            fields["point_stat"],
            fields["model_context_window"],
        )
        if key not in index or r["timestamp"] > index[key]["timestamp"]:
            index[key] = r
    assert len(checkpoints) <= 1, f"Mixed checkpoints in one grid dir: {checkpoints}"
    return index, (checkpoints.pop() if checkpoints else "<none>")


def get(
    index: dict[Key, dict],
    slug: str,
    strategy: str,
    k: int,
    point_stat: str,
    window: int | None = None,
) -> dict | None:
    return index.get((slug, strategy, k, point_stat, window))


def rmse_of(result: dict, split: str) -> float:
    return float(result[split]["rmse"])


def std_of(result: dict, split: str) -> float | None:
    value = result[split].get("rmse_std_seeds")
    return None if value is None else float(value)


def fmt_split_pair(result: dict | None) -> str:
    if result is None:
        return "—"
    parts = []
    for split in SPLITS:
        rmse = rmse_of(result, split)
        std = std_of(result, split)
        parts.append(f"{rmse:.2f}±{std:.2f}" if std is not None else f"{rmse:.2f}")
    return " / ".join(parts)


def _assert_same_example_ids(a: dict, b: dict, label: str) -> None:
    assert a["seeds"] == b["seeds"], f"Seed mismatch: {label}"
    for sa, sb in zip(a["per_seed"], b["per_seed"]):
        assert sa["example_ids"] == sb["example_ids"], (
            f"example_ids differ: {label} seed {sa['seed']}"
        )


def assert_twin_example_ids(index: dict[Key, dict]) -> int:
    """Hard-assert example-id equality across decoding/base-ft/window twins."""
    checked = 0
    for key, result in index.items():
        slug, strategy, k, point_stat, window = key
        if strategy == "zeroshot":
            continue
        # decoding twin: mean vs median
        if point_stat != "median":
            twin = index.get((slug, strategy, k, "median", window))
            if twin is not None:
                _assert_same_example_ids(result, twin, f"{key} vs median twin")
                checked += 1
        # base/ft twin (full window only)
        if slug == FINETUNED_SLUG and window is None:
            twin = index.get((BASE_SLUG, strategy, k, point_stat, None))
            if twin is not None:
                _assert_same_example_ids(result, twin, f"{key} vs base twin")
                checked += 1
        # window twin: win512 vs full
        if window is not None:
            twin = index.get((slug, strategy, k, point_stat, None))
            if twin is not None:
                _assert_same_example_ids(result, twin, f"{key} vs full-window twin")
                checked += 1
    return checked


def best_legit(
    index: dict[Key, dict], slug: str, point_stat: str
) -> tuple[tuple[str, int], dict] | None:
    """Best legit ICL config (oracle excluded) by ID RMSE for one model."""
    candidates = [
        ((s, k), r)
        for s, k in LEGIT_ICL
        if (r := get(index, slug, s, k, point_stat)) is not None
    ]
    return min(candidates, key=lambda cr: rmse_of(cr[1], "in_distribution"), default=None)


########################################################
# Comparison plumbing (same shape as the v4/v5 analyzers)
########################################################


def fmt_comparison(label: str, comp: PairedComparison) -> str:
    wilcoxon = "n/a" if comp.wilcoxon_p is None else f"{comp.wilcoxon_p:.3f}"
    return (
        f"| {label} | {SPLIT_SHORT[comp.split]} | {comp.rmse_a:.2f} | {comp.rmse_b:.2f} | "
        f"{comp.rmse_diff:+.2f} | [{comp.bootstrap_ci_low:.2f}, {comp.bootstrap_ci_high:.2f}] | "
        f"{comp.bootstrap_p:.3f} | {wilcoxon} |"
    )


COMPARISON_HEADER = (
    "\n| A vs B | split | RMSE A | RMSE B | Δ(A−B) | 95% CI | p_boot | p_wilcoxon |\n"
    "|---|---|---|---|---|---|---|---|"
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


########################################################
# Report blocks
########################################################


def make_headline(index: dict[Key, dict], checkpoint: str) -> list[str]:
    lines = [
        "# ICL × finetuning — Phase 6 results (does adaptation stack?)",
        "",
        f"Finetuned model: Chronos-2 + BilinearLoRA (r=8) with operating-param "
        f"conditioning, self-trained with Severin's exact notebook recipe "
        f"(`finetuning/chronos2/train_bilinear.py`; checkpoint `{checkpoint}`, "
        f"recorded in every finetuned cell's config — Severin's "
        f"`lora_weights.pt` can be swapped in for a re-run). Protocol: t266, "
        f"fixed 245-trace pool, flat concat + SHARED scaling, context 80, "
        f"prediction 64, tail 80. Base cells are bf16 (continuity with "
        f"v3–v5), finetuned cells fp32 (training numerics). Finetuned "
        f"forwards are conditioned on the QUERY's raw operating params "
        f"([shat, q, rlt, rln]) via `ConditionRegistry`. All cells from ONE "
        f"grid run (`results/few_shot_v6_finetuned/`); identical example "
        f"sets across decoding/base-ft/window twins (hard-asserted).",
    ]
    for point_stat in POINT_STATS:
        title = "headline (mean decoding — the v5 Chronos-2 default)" if point_stat == "mean" else "appendix (median decoding)"
        lines.append(f"\n## The 2×2 — {title}\n")
        lines.append("| model | k=0 (zero-shot) | best legit ICL | config |")
        lines.append("|---|---|---|---|")
        for slug, label in ((BASE_SLUG, "base"), (FINETUNED_SLUG, "finetuned")):
            z = get(index, slug, "zeroshot", 0, point_stat)
            best = best_legit(index, slug, point_stat)
            config_label = f"{best[0][0]} k={best[0][1]}" if best else "—"
            lines.append(
                f"| {label} | {fmt_split_pair(z)} | "
                f"{fmt_split_pair(best[1] if best else None)} | {config_label} |"
            )
        lines.append("\nCells are `ID / OOD` tail RMSE (±std over seeds where multi-seed).")
    return lines


def make_full_table(index: dict[Key, dict]) -> list[str]:
    lines = ["\n## Full grid — base vs finetuned per config\n"]
    for point_stat in POINT_STATS:
        lines.append(f"\n### Decoding: {point_stat}\n")
        lines.append("| config | base | finetuned | Δft−base (ID) | Δ (OOD) |")
        lines.append("|---|---|---|---|---|")
        for strategy, k in GRID_CONFIGS:
            base = get(index, BASE_SLUG, strategy, k, point_stat)
            ft = get(index, FINETUNED_SLUG, strategy, k, point_stat)
            label = f"{strategy} k={k}" + (" (cheats)" if strategy == "oracle_tail" else "")
            cells = [label, fmt_split_pair(base), fmt_split_pair(ft)]
            for split in SPLITS:
                if base is not None and ft is not None:
                    cells.append(f"{rmse_of(ft, split) - rmse_of(base, split):+.2f}")
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def make_paired_block(index: dict[Key, dict]) -> list[str]:
    lines = [
        "\n## Paired comparisons",
        "",
        "Δ = RMSE(A) − RMSE(B); negative favours A. CI/p from a 10k paired "
        "bootstrap over traces (Wilcoxon p floors 0.031 ID / 0.0625 OOD); "
        "`[trace_seed]` rows for the 20-seed random cells.",
        COMPARISON_HEADER,
    ]
    for point_stat in POINT_STATS:
        ft0 = get(index, FINETUNED_SLUG, "zeroshot", 0, point_stat)
        base0 = get(index, BASE_SLUG, "zeroshot", 0, point_stat)
        if ft0 is not None and base0 is not None:
            lines.extend(
                comparison_rows(f"[{point_stat}] ft k0 vs base k0", ft0, base0)
            )
        for strategy, k in GRID_CONFIGS:
            if (strategy, k) == ("zeroshot", 0):
                continue
            base = get(index, BASE_SLUG, strategy, k, point_stat)
            ft = get(index, FINETUNED_SLUG, strategy, k, point_stat)
            tag = f"[{point_stat}] {strategy} k={k}"
            if ft is not None and base is not None:
                lines.extend(comparison_rows(f"{tag}: ft vs base", ft, base))
            if ft is not None and ft0 is not None and len(ft["seeds"]) == len(ft0["seeds"]):
                lines.extend(comparison_rows(f"{tag}: ft+ICL vs ft k0", ft, ft0))
            if base is not None and base0 is not None and len(base["seeds"]) == len(base0["seeds"]):
                lines.extend(comparison_rows(f"{tag}: base+ICL vs base k0", base, base0))
    return lines


def make_window_block(index: dict[Key, dict]) -> list[str]:
    lines = [
        f"\n## Context window: full (8192) vs the {FT_TRAIN_CONTEXT} training window",
        "",
        f"The finetuned model trained exclusively on {FT_TRAIN_CONTEXT}-wide "
        f"windows; a k=10 ICL stream is 3550 steps. `pipeline.predict` "
        f"silently clamps the stream down to `chronos_config.context_length`, "
        f"so the win{FT_TRAIN_CONTEXT} cells see only the LAST "
        f"{FT_TRAIN_CONTEXT} steps (tail of the last example + query). "
        f"k=0 streams (≤266) fit either window — those cells are "
        f"window-invariant by construction (asserted in smoke F5) and were "
        f"not duplicated.",
        "",
        "| config | decoding | full window | win512 | Δwin512−full (ID) | Δ (OOD) |",
        "|---|---|---|---|---|---|",
    ]
    comparisons: list[str] = []
    for strategy, k in WINDOW_CONFIGS:
        for point_stat in POINT_STATS:
            full = get(index, FINETUNED_SLUG, strategy, k, point_stat, None)
            win = get(index, FINETUNED_SLUG, strategy, k, point_stat, FT_TRAIN_CONTEXT)
            if win is None:
                continue
            deltas = [
                f"{rmse_of(win, split) - rmse_of(full, split):+.2f}" if full else "—"
                for split in SPLITS
            ]
            lines.append(
                f"| {strategy} k={k} | {point_stat} | {fmt_split_pair(full)} | "
                f"{fmt_split_pair(win)} | {deltas[0]} | {deltas[1]} |"
            )
            if full is not None:
                comparisons.extend(
                    comparison_rows(
                        f"[{point_stat}] {strategy} k={k}: win512 vs full", win, full
                    )
                )
    lines.append(COMPARISON_HEADER)
    lines.extend(comparisons)
    return lines


def make_anchor_block(anchor: dict | None) -> list[str]:
    lines = ["\n## Severin-protocol anchor (notebook eval, both metrics)", ""]
    if anchor is None:
        lines.append("`severin_anchor.json` not found — anchor stage not run.")
        return lines
    sev = anchor["metrics_severin_headminus80"]
    tail = anchor["metrics_tail80"]
    lines.append(
        "The five chronos2 finetuning notebooks score `mean(x[:-80])` — the "
        "mean over everything EXCEPT the tail, *including the 80 copied "
        "ground-truth context steps* — while our tables, the GyroSwin paper, "
        "and the repo's own TimesFM runner (`experiments/model.py`) use the "
        "proper `mean(x[-80:])` tail. The README's Chronos-2 finetuning rows "
        "(BilinearLoRA **13.83 ID / 4.86 OOD**) are therefore on a "
        "different, easier metric. Both metrics below are computed from the "
        "SAME forecasts of OUR checkpoint under his exact protocol (raw "
        "forward, NaN-padded 512 window, 21-quantile median, autoregressive "
        "from step 80, `[0::3]` traces):",
        )
    lines.append("")
    lines.append("| metric | ID RMSE | OOD RMSE | comparable to |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| his `mean(x[:-80])` | {sev['id']['rmse']:.2f} ± {sev['id']['standard_error']:.2f} "
        f"| {sev['ood']['rmse']:.2f} ± {sev['ood']['standard_error']:.2f} "
        f"| README finetuning rows (13.83 / 4.86) |"
    )
    lines.append(
        f"| honest `mean(x[-80:])` | {tail['id']['rmse']:.2f} ± {tail['id']['standard_error']:.2f} "
        f"| {tail['ood']['rmse']:.2f} ± {tail['ood']['standard_error']:.2f} "
        f"| our tables / GyroSwin / TimesFM runner |"
    )
    lines.append(
        "\nOur his-metric number differs from the README's 13.83 because it "
        "is a different RUN (self-trained checkpoint, MPS vs his CUDA, "
        "different RNG) — the distance is reported, not asserted. The "
        "harness-protocol finetuned rungs in the ladder below are the "
        "numbers to carry forward."
    )
    if "vs_severin" in anchor:
        comp = anchor["vs_severin"]
        rels = [v for split in comp["per_trace_rel_l2"].values() for v in split.values()]
        lines.append(
            f"\nPer-trace drift vs Severin's `benchmark_results.json` "
            f"(`{comp['path']}`): max rel L2 {max(rels):.3f}, "
            f"mean {float(np.mean(rels)):.3f}."
        )
    return lines


def make_bridge_block(index: dict[Key, dict], v5_dir: Path) -> list[str]:
    """Per-trace drift of the v6 base cells vs their v5 counterparts."""
    lines = [
        "\n## v5 bridge — base cells re-run vs `results/few_shot_v5_decoding`",
        "",
        "Same machine, same code path; MPS is not guaranteed bit-deterministic "
        "across process runs, so drift is REPORTED (bit-equality expected on "
        "this machine).",
        "",
        "| config | decoding | max rel Δ pred_tail_mean | bit-equal traces |",
        "|---|---|---|---|",
    ]
    v5_results = load_results(v5_dir)
    v5_index: dict[tuple[str, int, str], dict] = {}
    for r in v5_results:
        config = r.get("config", {})
        if config.get("model_slug") != BASE_SLUG:
            continue
        method = r.get("method", "")
        if "__" not in method:
            continue
        strategy = method[len(BASE_SLUG.replace("/", "_") + "_"):].rsplit("__", 1)[0]
        key = (strategy, int(config["k_shot"]), config.get("point_stat", "median"))
        if key not in v5_index or r["timestamp"] > v5_index[key]["timestamp"]:
            v5_index[key] = r
    for strategy, k in BRIDGE_CONFIGS:
        for point_stat in POINT_STATS:
            v6 = get(index, BASE_SLUG, strategy, k, point_stat)
            v5 = v5_index.get((strategy, k, point_stat))
            if v6 is None or v5 is None:
                lines.append(f"| {strategy} k={k} | {point_stat} | — (missing) | — |")
                continue
            max_rel, n_bit, n_total = 0.0, 0, 0
            for s6, s5 in zip(v6["per_seed"], v5["per_seed"]):
                t6 = {tr["trace_key"]: tr["pred_tail_mean"] for tr in s6["per_trace"]}
                t5 = {tr["trace_key"]: tr["pred_tail_mean"] for tr in s5["per_trace"]}
                for trace_key in t6:
                    if trace_key not in t5:
                        continue
                    n_total += 1
                    n_bit += t6[trace_key] == t5[trace_key]
                    rel = abs(t6[trace_key] - t5[trace_key]) / max(1.0, abs(t5[trace_key]))
                    max_rel = max(max_rel, rel)
            lines.append(
                f"| {strategy} k={k} | {point_stat} | {max_rel:.2e} | "
                f"{n_bit}/{n_total} |"
            )
    return lines


########################################################
# Ladder (regenerated with honest finetuned rungs)
########################################################

#: Cross-run references for the regenerated ladder. Severin's finetuning
#: rows are scored with mean(x[:-80]) — head mean INCLUDING the 80 copied
#: ground-truth context steps — and are NOT comparable to the tail rungs.
LADDER_REFERENCES: list[dict] = [
    {
        "label": "GPR (paper baseline)",
        "id": 43.82,
        "ood": 59.28,
        "kind": "reference",
    },
    {
        "label": "GyroSwin-1B (paper)",
        "id": 18.35,
        "ood": 26.43,
        "kind": "reference",
    },
    {
        "label": "Severin's BilinearLoRA (README; [:-80] metric — NOT comparable)",
        "id": 13.83,
        "ood": 4.86,
        "kind": "incomparable",
    },
]


def ladder_rows(index: dict[Key, dict], anchor: dict | None) -> list[dict]:
    """Measured v6 rungs (mean decoding) + annotated references."""
    rows: list[dict] = [LADDER_REFERENCES[0]]

    def add(label: str, result: dict | None, kind: str = "v6") -> None:
        if result is not None:
            rows.append(
                {
                    "label": label,
                    "id": rmse_of(result, "in_distribution"),
                    "ood": rmse_of(result, "out_of_distribution"),
                    "kind": kind,
                }
            )

    add("Chronos-2 zero-shot (base)", get(index, BASE_SLUG, "zeroshot", 0, "mean"))
    base_best = best_legit(index, BASE_SLUG, "mean")
    if base_best:
        add(
            f"Chronos-2 ICL (base, {base_best[0][0]} k={base_best[0][1]})",
            base_best[1],
        )
    add(
        "Chronos-2 BilinearLoRA finetuned, k=0 (ours, harness protocol)",
        get(index, FINETUNED_SLUG, "zeroshot", 0, "mean"),
    )
    ft_best = best_legit(index, FINETUNED_SLUG, "mean")
    if ft_best:
        add(
            f"finetuned + ICL ({ft_best[0][0]} k={ft_best[0][1]})",
            ft_best[1],
        )
    win_cells = [
        ((s, k), r)
        for s, k in WINDOW_CONFIGS
        if (r := get(index, FINETUNED_SLUG, s, k, "mean", FT_TRAIN_CONTEXT)) is not None
    ]
    if win_cells:
        (s, k), r = min(win_cells, key=lambda cr: rmse_of(cr[1], "in_distribution"))
        add(f"finetuned + ICL @ {FT_TRAIN_CONTEXT} training window ({s} k={k})", r)
    if anchor is not None:
        tail = anchor["metrics_tail80"]
        rows.append(
            {
                "label": "finetuned, Severin's rollout protocol (honest [-80:] rescore)",
                "id": tail["id"]["rmse"],
                "ood": tail["ood"]["rmse"],
                "kind": "anchor",
            }
        )
    rows.append(LADDER_REFERENCES[1])  # GyroSwin-1B
    rows.append(LADDER_REFERENCES[2])  # Severin's [:-80] row, annotated
    return rows


def fig_ladder(rows: list[dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = {
        "reference": "0.65",
        "v6": "tab:blue",
        "anchor": "tab:cyan",
        "incomparable": "mistyrose",
    }
    for ax, split_key, title in ((axes[0], "id", "ID"), (axes[1], "ood", "OOD")):
        labels = [row["label"] for row in rows]
        values = [row[split_key] for row in rows]
        bar_colors = [colors[row["kind"]] for row in rows]
        hatches = ["xx" if row["kind"] == "incomparable" else "" for row in rows]
        y = np.arange(len(rows))
        bars = ax.barh(y, values, color=bar_colors, height=0.62)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for yi, value in zip(y, values):
            ax.text(value + 0.5, yi, f"{value:.1f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("tail RMSE")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
        plt.Rectangle((0, 0), 1, 1, color="tab:cyan"),
        plt.Rectangle((0, 0), 1, 1, color="0.65"),
        plt.Rectangle((0, 0), 1, 1, color="mistyrose", hatch="xx"),
    ]
    fig.legend(
        handles,
        [
            "v6 grid (harness, tail RMSE)",
            "Severin-protocol rollout, honest rescore",
            "cross-run reference",
            "[:-80] metric — not comparable",
        ],
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Adaptation ladder (regenerated): zero-shot → ICL → finetuned → finetuned + ICL",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    path = out_dir / "adaptation_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_ladder_table(rows: list[dict]) -> list[str]:
    lines = [
        "\n## Adaptation ladder (regenerated — honest finetuned rung)",
        "",
        "All blue rungs are measured v6 cells (mean decoding, harness tail "
        "RMSE on the same 11 traces). The previous ladder's finetuned rung "
        "cited Severin's README 13.83 ID — that number is on the notebooks' "
        "`mean(x[:-80])` metric (includes the copied context) and is kept "
        "only as an annotated, non-comparable reference.",
        "",
        "| rung | ID | OOD | basis |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['id']:.2f} | {row['ood']:.2f} | {row['kind']} |"
        )
    return lines


########################################################
# Synergy figure
########################################################


def fig_synergy(index: dict[Key, dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    configs = list(GRID_CONFIGS)
    for ax, split in zip(axes, SPLITS):
        for xi, (strategy, k) in enumerate(configs):
            base = get(index, BASE_SLUG, strategy, k, "mean")
            ft = get(index, FINETUNED_SLUG, strategy, k, "mean")
            if base is None or ft is None:
                continue
            y_base, y_ft = rmse_of(base, split), rmse_of(ft, split)
            color = "tab:red" if strategy == "oracle_tail" else "tab:blue"
            ax.annotate(
                "",
                xy=(xi, y_ft),
                xytext=(xi, y_base),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4},
            )
            ax.plot(xi, y_base, "o", mfc="white", mec=color, ms=7, zorder=3)
            ax.plot(xi, y_ft, "o", color=color, ms=7, zorder=3)
        ax.set_xticks(np.arange(len(configs)))
        ax.set_xticklabels(
            [f"{s}\nk={k}" for s, k in configs], fontsize=8, rotation=0
        )
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(SPLIT_SHORT[split], fontsize=11)
    axes[0].set_ylabel("tail RMSE")
    handles = [
        plt.Line2D([], [], marker="o", mfc="white", mec="0.2", color="none", label="base (bf16)"),
        plt.Line2D([], [], marker="o", color="0.2", ls="", label="finetuned (fp32, OP-conditioned)"),
        plt.Line2D([], [], color="tab:red", label="oracle_tail (cheats)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle(
        "ICL × finetuning: base → finetuned per config (mean decoding, shared scaling, v6 grid)",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path = out_dir / "finetuned_synergy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


########################################################
# Verdict (data-driven scaffold; prose finalized with the numbers)
########################################################


def make_verdict(index: dict[Key, dict]) -> list[str]:
    lines = ["\n## Verdict — does adaptation stack?", ""]
    rows = []
    for point_stat in POINT_STATS:
        base0 = get(index, BASE_SLUG, "zeroshot", 0, point_stat)
        ft0 = get(index, FINETUNED_SLUG, "zeroshot", 0, point_stat)
        base_best = best_legit(index, BASE_SLUG, point_stat)
        ft_best = best_legit(index, FINETUNED_SLUG, point_stat)
        if None in (base0, ft0, base_best, ft_best):
            continue
        for split in SPLITS:
            rows.append(
                {
                    "point_stat": point_stat,
                    "split": SPLIT_SHORT[split],
                    "base0": rmse_of(base0, split),
                    "ft0": rmse_of(ft0, split),
                    "base_icl": rmse_of(base_best[1], split),
                    "ft_icl": rmse_of(ft_best[1], split),
                }
            )
    if rows:
        lines.append(
            "| decoding | split | base k0 | ft k0 | base+ICL | ft+ICL | "
            "finetuning gain at k0 | finetuning gain at best-k | ICL gain on ft |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['point_stat']} | {r['split']} | {r['base0']:.2f} | {r['ft0']:.2f} | "
                f"{r['base_icl']:.2f} | {r['ft_icl']:.2f} | "
                f"{r['ft0'] - r['base0']:+.2f} | {r['ft_icl'] - r['base_icl']:+.2f} | "
                f"{r['ft_icl'] - r['ft0']:+.2f} |"
            )
    lines.append(VERDICT)
    return lines


#: Finalized after the grid run (data-driven scaffold above stays).
VERDICT = """
**Verdict — adaptation stacks, but only through retrieval quality, and the
training window matters.** (1) **Finetuning dominates the ladder**: ft k=0
(22.20 ID / 34.10 OOD, mean decoding) beats base k=0 (89.51 / 67.94;
bootstrap-significant both splits) — and already beats the best BASE ICL
cell (27.06 ID). (2) **Legit retrieval-ICL adds a further ID gain on top**:
22.20 → 18.62 (mmr_euclid k=5), and 15.63 under the 512 TRAINING window —
the project's best legitimate training-free-at-inference number (previous
best: Bolt 22.63). With n=6 ID traces the marginal gain is not individually
significant (CI [−16.4, +16.8]); the direction is consistent across all
four mmr cells and both windows. (3) **The oracle proves ICL capacity
SURVIVED finetuning**: oracle_tail k=10 stacks significantly on the ft
model (9.39 ID, p=0.043; 10.89 OOD, p=0.002 vs ft k0) and the ft model
exploits oracle examples better than the base does (−12.2 ID, p=0.032) —
the bottleneck is retrieval quality, not the model's in-context ability.
(4) **Bad examples destroy the finetuned advantage**: with random k=10
examples ft ≈ base exactly (40.02 vs 39.99 ID; +0.03, n.s. even at
trace_seed resolution) — random examples drag the ft model from 22.20 UP to
~39, the same level they pull the base DOWN to from 89.51. Once the model
is finetuned, example quality is no longer optional. (5) **OOD is
finetuning's story alone**: 67.94 → 34.10 at k=0; no legit ICL config
improves it further (mmr +1.9..+3.1); only the window clamp mildly helps
(32.34). (6) **The 512 training window beats the full 8192 window in all 8
paired cells** (−3.0 to −10.4; p 0.08–0.55): long concat streams are
out-of-distribution for a model finetuned exclusively on 512-step windows.
Under the clamp, k=5 and k=10 are bit-identical (both reduce to the same
last-512 stream ≈ the tail of the final example + query) — effective ICL
for the finetuned model is "one well-chosen example tail inside the
training window". RAF's retrieval+finetuning synergy claim is qualitatively
supported on ID; 6 traces cannot make the marginal gain significant.

Caveats: self-trained checkpoint (recipe-faithful — the recipe's
load_best_model_at_end picked step 200 of 4000 under its noisy 25-series
random-cutoff eval; train loss fell monotonically 5.68 → 2.65); base bf16
vs ft fp32; one checkpoint, one training run. Severin's weights swap in via
`--checkpoint` for a minutes-long re-run.
""".rstrip()


NOTE_FOR_SEVERIN = """
## Note for Severin — the chronos2 notebooks' benchmark metric

All five chronos2 finetuning notebooks (`chronos2_{bilinear,lora,full,
oss_bilinear,rss_bilinear}.ipynb`, eval cells) score

```python
np.mean(flux_data.energy_flux[:-80])   # and np.mean(forecast[:-80])
```

— the mean over everything EXCEPT the last 80 steps, which *includes the
80 ground-truth context steps the forecast copies verbatim* (the rollout
starts from `START_IDX = 80`). Your own TimesFM runner
(`experiments/model.py`) uses the proper tail `[-80:]`, as do the GyroSwin
paper and our few-shot tables. The README's Chronos-2 finetuning rows are
therefore on a different, easier metric than every number they are
compared against.

Measured effect (our self-trained BilinearLoRA checkpoint, your exact
protocol, SAME forecasts, only the scoring window changed): `[:-80]` gives
ID 15.72 / OOD 6.03 — close to your published 13.83 / 4.86 — while the
honest `[-80:]` rescore gives **ID 17.51 / OOD 40.64**. The dramatic OOD
numbers in the README's chronos2 rows are largely the copied-context
artifact; the ID numbers are only mildly inflated. The TimesFM rows are
unaffected. Re-scoring your saved `benchmark_results.json` files takes one
line per file (the forecasts are stored full-length); happy to share
`severin_anchor_eval` (`benchmarking/few_shot/finetuned.py`), which
computes both metrics side by side.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Phase-6 finetuned-ICL grid")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--v5-dir", type=Path, default=V5_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    index, checkpoint = build_index(load_results(args.results_dir))
    assert index, f"No finetuned-grid results in {args.results_dir}"
    n_twins = assert_twin_example_ids(index)
    print(f"✓ example_ids identical across {n_twins} twin pairs; checkpoint {checkpoint}")

    anchor_path = args.results_dir / "severin_anchor.json"
    anchor = json.load(open(anchor_path)) if anchor_path.exists() else None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = make_headline(index, checkpoint)
    synergy_path = fig_synergy(index, args.out_dir)
    lines.append(f"\n![{synergy_path.stem}]({synergy_path.name})")
    lines.extend(make_full_table(index))
    lines.extend(make_paired_block(index))
    lines.extend(make_window_block(index))
    lines.extend(make_anchor_block(anchor))
    rows = ladder_rows(index, anchor)
    ladder_path = fig_ladder(rows, args.out_dir)
    lines.extend(make_ladder_table(rows))
    lines.append(f"\n![{ladder_path.stem}]({ladder_path.name})")
    lines.extend(make_bridge_block(index, args.v5_dir))
    lines.extend(make_verdict(index))
    lines.append("\n" + NOTE_FOR_SEVERIN)

    markdown = "\n".join(lines) + "\n"
    table_path = args.out_dir / "finetuned_icl_table.md"
    table_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {table_path}")
    print(f"Wrote {synergy_path}")
    print(f"Wrote {ladder_path}")


if __name__ == "__main__":
    main()
