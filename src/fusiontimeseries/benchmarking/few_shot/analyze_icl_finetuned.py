"""Analysis of the Phase-9 in-context-finetuning (ICF) eval grid.

Loads ``results/few_shot_v9_icl/`` (the two ICF checkpoints: ``level`` and the
``random`` control) and ``results/few_shot_v6_finetuned/`` (the v6 single-trace
finetuned model, the inherited-ICL baseline) and answers, in
``docs/results/fewshot/``:

- ``icl_finetuning_table.md`` — the three-way headline (ICF-level vs ICF-random
  vs v6-ft), the full per-config table (mean headline, median appendix), paired
  comparisons (level vs random = "did it learn to USE demos?"; each checkpoint's
  ICL gain vs its own k=0; ICF-level vs v6-ft = "does demonstration-training
  beat inherited ICL?"), a k-curve, the ICF ladder rung, and the verdict;
- ``icl_finetuning_kcurve.png`` — ID/OOD tail RMSE vs k for the three models,
  per retrieval strategy (mean decoding).

The ICF models are evaluated at their 2048 training window; the v6 ft model at
its full (8192) window — the comparison is each finetuned model at its own
natural window (noted in captions). n=6 ID / 5 OOD traces ⇒ report CIs, do not
over-claim significance.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_icl_finetuned
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_icl_finetuned --self-test
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fusiontimeseries.benchmarking.few_shot.analyze_finetuned import (
    COMPARISON_HEADER,
    SPLIT_SHORT,
    SPLITS,
    build_index as build_v6_index,
    comparison_rows,
    fmt_split_pair,
    get as v6_get,
    parse_variant,
    rmse_of,
)
from fusiontimeseries.benchmarking.few_shot.finetuned import FINETUNED_SLUG
from fusiontimeseries.benchmarking.few_shot.harness import load_results
from fusiontimeseries.benchmarking.few_shot.run_icl_finetuned import (
    ICF_EVAL_WINDOW,
    ICF_SLUGS,
    ICL_CONFIGS,
    RANDOM_K,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_V9_DIR: Path = REPO_ROOT / "results" / "few_shot_v9_icl"
DEFAULT_V6_DIR: Path = REPO_ROOT / "results" / "few_shot_v6_finetuned"
DEFAULT_OUT_DIR: Path = REPO_ROOT / "docs" / "results" / "fewshot"

POINT_STATS: tuple[str, ...] = ("mean", "median")
MODES: tuple[str, ...] = ("level", "random")
SLUG_TO_MODE: dict[str, str] = {slug: mode for mode, slug in ICF_SLUGS.items()}

#: All ICF grid configs in display order.
GRID_CONFIGS: tuple[tuple[str, int], ...] = (
    ("zeroshot", 0),
    *ICL_CONFIGS,
    ("random", RANDOM_K),
)
LEGIT_ICL: tuple[tuple[str, int], ...] = tuple(
    (s, k) for s, k in ICL_CONFIGS if s != "oracle_tail"
)

#: ICF index key: (mode, strategy, k, point_stat). Window is the constant
#: ICF_EVAL_WINDOW for every cell, so it is not part of the key.
V9Key = tuple[str, str, int, str]


########################################################
# Indexing
########################################################


def build_v9_index(results: list[dict]) -> tuple[dict[V9Key, dict], dict[str, str]]:
    """Index the ICF cells; latest timestamp wins; one checkpoint id per mode."""
    index: dict[V9Key, dict] = {}
    checkpoints: dict[str, set[str]] = {}
    for r in results:
        config = r.get("config", {})
        slug = config.get("model_slug")
        mode = SLUG_TO_MODE.get(slug)
        if mode is None:
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
        window = fields["model_context_window"]
        assert window == ICF_EVAL_WINDOW, (
            f"{method}: ICF cell window {window} != {ICF_EVAL_WINDOW}"
        )
        assert config.get("checkpoint"), f"{method}: ICF cell without checkpoint id"
        checkpoints.setdefault(mode, set()).add(config["checkpoint"])
        key: V9Key = (mode, strategy, int(config["k_shot"]), fields["point_stat"])
        if key not in index or r["timestamp"] > index[key]["timestamp"]:
            index[key] = r
    resolved: dict[str, str] = {}
    for mode, cks in checkpoints.items():
        assert len(cks) == 1, f"Mixed checkpoints for ICF-{mode}: {cks}"
        resolved[mode] = cks.pop()
    return index, resolved


def v9_get(
    index: dict[V9Key, dict], mode: str, strategy: str, k: int, point_stat: str
) -> dict | None:
    return index.get((mode, strategy, k, point_stat))


def best_legit_v9(
    index: dict[V9Key, dict], mode: str, point_stat: str
) -> tuple[tuple[str, int], dict] | None:
    """Best legit ICL config (oracle excluded) by ID RMSE for one ICF mode."""
    candidates = [
        ((s, k), r)
        for s, k in LEGIT_ICL
        if (r := v9_get(index, mode, s, k, point_stat)) is not None
    ]
    return min(
        candidates, key=lambda cr: rmse_of(cr[1], "in_distribution"), default=None
    )


def best_legit_v6(
    v6_index: dict, point_stat: str
) -> tuple[tuple[str, int], dict] | None:
    """Best legit ICL config for the v6 single-trace ft model (full window)."""
    candidates = [
        ((s, k), r)
        for s, k in LEGIT_ICL
        if (r := v6_get(v6_index, FINETUNED_SLUG, s, k, point_stat, None)) is not None
    ]
    return min(
        candidates, key=lambda cr: rmse_of(cr[1], "in_distribution"), default=None
    )


########################################################
# Report blocks
########################################################


def make_headline(
    v9: dict[V9Key, dict], v6_index: dict, checkpoints: dict[str, str]
) -> list[str]:
    lines = [
        "# In-context finetuning — Phase 9 results "
        "(does training on demonstrations beat inherited ICL?)",
        "",
        "The v6 finetuned model's in-context ability is INHERITED from base "
        "pretraining — it was finetuned on single traces. Phase 9 trains the "
        "SAME BilinearLoRA recipe ON multi-example ICL concatenations "
        "(`Chronos2ICLDataset`, window 2048, k∈{1,3,5} per sample, query-only "
        "conditioning) with two retrieval modes:",
        "",
        f"- **ICF-level** (`{checkpoints.get('level', '<none>')}`): demos "
        "retrieved by context level during training (train≡test `ctx_level`).",
        f"- **ICF-random** (`{checkpoints.get('random', '<none>')}`): demos "
        "sampled at random — the control that probes whether the model learned "
        "to USE level-matched demos or merely tolerate demonstrations.",
        "",
        "Both are evaluated at their 2048 training window through the frozen "
        "shared-scaling rollout; the v6 single-trace ft model (the "
        "inherited-ICL baseline) is shown at its full (8192) window. Each "
        "finetuned model is at its own natural window. n=6 ID / 5 OOD traces.",
    ]
    for point_stat in POINT_STATS:
        title = "headline (mean decoding)" if point_stat == "mean" else "appendix (median decoding)"
        lines.append(f"\n## Three-way — {title}\n")
        lines.append("| model | k=0 (zero-shot) | best legit ICL | config |")
        lines.append("|---|---|---|---|")
        for mode in MODES:
            z = v9_get(v9, mode, "zeroshot", 0, point_stat)
            best = best_legit_v9(v9, mode, point_stat)
            config_label = f"{best[0][0]} k={best[0][1]}" if best else "—"
            lines.append(
                f"| ICF-{mode} | {fmt_split_pair(z)} | "
                f"{fmt_split_pair(best[1] if best else None)} | {config_label} |"
            )
        z6 = v6_get(v6_index, FINETUNED_SLUG, "zeroshot", 0, point_stat, None)
        best6 = best_legit_v6(v6_index, point_stat)
        config6 = f"{best6[0][0]} k={best6[0][1]}" if best6 else "—"
        lines.append(
            f"| v6 single-trace ft (8192 win) | {fmt_split_pair(z6)} | "
            f"{fmt_split_pair(best6[1] if best6 else None)} | {config6} |"
        )
        lines.append("\nCells are `ID / OOD` tail RMSE (±std over seeds where multi-seed).")
    return lines


def make_full_table(v9: dict[V9Key, dict], v6_index: dict) -> list[str]:
    lines = ["\n## Full grid — ICF-level vs ICF-random vs v6 ft per config\n"]
    for point_stat in POINT_STATS:
        lines.append(f"\n### Decoding: {point_stat}\n")
        lines.append(
            "| config | ICF-level | ICF-random | v6 ft (8192) | "
            "Δ level−random (ID) | Δ level−v6 (ID) |"
        )
        lines.append("|---|---|---|---|---|---|")
        for strategy, k in GRID_CONFIGS:
            lvl = v9_get(v9, "level", strategy, k, point_stat)
            rnd = v9_get(v9, "random", strategy, k, point_stat)
            v6 = v6_get(v6_index, FINETUNED_SLUG, strategy, k, point_stat, None)
            label = f"{strategy} k={k}" + (" (cheats)" if strategy == "oracle_tail" else "")
            cells = [label, fmt_split_pair(lvl), fmt_split_pair(rnd), fmt_split_pair(v6)]
            split = "in_distribution"
            cells.append(
                f"{rmse_of(lvl, split) - rmse_of(rnd, split):+.2f}"
                if lvl is not None and rnd is not None
                else "—"
            )
            cells.append(
                f"{rmse_of(lvl, split) - rmse_of(v6, split):+.2f}"
                if lvl is not None and v6 is not None
                else "—"
            )
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def make_paired_block(v9: dict[V9Key, dict], v6_index: dict) -> list[str]:
    lines = [
        "\n## Paired comparisons",
        "",
        "Δ = RMSE(A) − RMSE(B); negative favours A. CI/p from a 10k paired "
        "bootstrap over traces; `[trace_seed]` rows for the 20-seed random "
        "cells. Three questions: **L vs R** (did ICF learn to USE level-matched "
        "demos — is the level checkpoint better than the random control at the "
        "same eval config?); **+ICL vs k0** (does ICL help each checkpoint over "
        "its own zero-shot?); **L vs v6** (does demonstration-training beat the "
        "v6 inherited-ICL model?).",
        COMPARISON_HEADER,
    ]
    for point_stat in POINT_STATS:
        lvl0 = v9_get(v9, "level", "zeroshot", 0, point_stat)
        rnd0 = v9_get(v9, "random", "zeroshot", 0, point_stat)
        if lvl0 is not None and rnd0 is not None:
            lines.extend(comparison_rows(f"[{point_stat}] k0: ICF-level vs ICF-random", lvl0, rnd0))
        for strategy, k in GRID_CONFIGS:
            if (strategy, k) == ("zeroshot", 0):
                continue
            lvl = v9_get(v9, "level", strategy, k, point_stat)
            rnd = v9_get(v9, "random", strategy, k, point_stat)
            v6 = v6_get(v6_index, FINETUNED_SLUG, strategy, k, point_stat, None)
            tag = f"[{point_stat}] {strategy} k={k}"
            if lvl is not None and rnd is not None:
                lines.extend(comparison_rows(f"{tag}: L vs R", lvl, rnd))
            if lvl is not None and lvl0 is not None and len(lvl["seeds"]) == len(lvl0["seeds"]):
                lines.extend(comparison_rows(f"{tag}: L+ICL vs L k0", lvl, lvl0))
            if rnd is not None and rnd0 is not None and len(rnd["seeds"]) == len(rnd0["seeds"]):
                lines.extend(comparison_rows(f"{tag}: R+ICL vs R k0", rnd, rnd0))
            if lvl is not None and v6 is not None and len(lvl["seeds"]) == len(v6["seeds"]):
                lines.extend(comparison_rows(f"{tag}: L vs v6 ft", lvl, v6))
    return lines


########################################################
# k-curve figure
########################################################


def fig_kcurve(v9: dict[V9Key, dict], v6_index: dict, out_dir: Path) -> Path:
    strategies = ["ctx_level", "mmr_euclid", "mmr_level"]
    ks = [0, 5, 10]
    fig, axes = plt.subplots(2, len(strategies), figsize=(4.2 * len(strategies), 7.6), sharex=True)
    series = [
        ("ICF-level", "tab:blue", lambda s, k: v9_get(v9, "level", s, k, "mean")),
        ("ICF-random", "tab:orange", lambda s, k: v9_get(v9, "random", s, k, "mean")),
        ("v6 ft (8192)", "tab:green", lambda s, k: v6_get(v6_index, FINETUNED_SLUG, s, k, "mean", None)),
    ]
    for col, strategy in enumerate(strategies):
        for row, split in enumerate(SPLITS):
            ax = axes[row, col]
            for label, color, getter in series:
                xs, ys = [], []
                for k in ks:
                    s = "zeroshot" if k == 0 else strategy
                    r = getter(s, 0) if k == 0 else getter(strategy, k)
                    if r is not None:
                        xs.append(k)
                        ys.append(rmse_of(r, split))
                if xs:
                    ax.plot(xs, ys, "o-", color=color, label=label, ms=6)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(strategy, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{SPLIT_SHORT[split]} tail RMSE")
            if row == 1:
                ax.set_xlabel("k (shots)")
                ax.set_xticks(ks)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        "In-context finetuning: ICL k-curve (mean decoding) — "
        "ICF-level vs ICF-random vs v6 single-trace ft",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = out_dir / "icl_finetuning_kcurve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


########################################################
# Ladder rung + verdict
########################################################


def make_ladder_block(v9: dict[V9Key, dict], v6_index: dict) -> list[str]:
    lines = [
        "\n## ICF rung on the adaptation ladder (mean decoding)",
        "",
        "Where the ICF checkpoints land relative to the v6 finetuned rungs "
        "(all tail RMSE, harness protocol). The v6 ft rungs are at the 8192 "
        "window; ICF rungs at the 2048 training window.",
        "",
        "| rung | ID | OOD |",
        "|---|---|---|",
    ]

    def row(label: str, r: dict | None) -> None:
        if r is not None:
            lines.append(
                f"| {label} | {rmse_of(r, 'in_distribution'):.2f} | "
                f"{rmse_of(r, 'out_of_distribution'):.2f} |"
            )

    row("v6 ft k=0 (single-trace, inherited ICL)",
        v6_get(v6_index, FINETUNED_SLUG, "zeroshot", 0, "mean", None))
    b6 = best_legit_v6(v6_index, "mean")
    if b6:
        row(f"v6 ft + ICL ({b6[0][0]} k={b6[0][1]})", b6[1])
    for mode in MODES:
        row(f"ICF-{mode} k=0", v9_get(v9, mode, "zeroshot", 0, "mean"))
        bm = best_legit_v9(v9, mode, "mean")
        if bm:
            row(f"ICF-{mode} + ICL ({bm[0][0]} k={bm[0][1]})", bm[1])
        oracle = v9_get(v9, mode, "oracle_tail", 10, "mean")
        row(f"ICF-{mode} + oracle k=10 (ceiling)", oracle)
    return lines


def make_verdict(v9: dict[V9Key, dict], v6_index: dict) -> list[str]:
    """Data-driven scaffold; prose VERDICT finalized after the run."""
    lines = ["\n## Verdict — did training on demonstrations help?", ""]
    rows = []
    for point_stat in POINT_STATS:
        for split in SPLITS:
            lvl0 = v9_get(v9, "level", "zeroshot", 0, point_stat)
            rnd0 = v9_get(v9, "random", "zeroshot", 0, point_stat)
            lvl_best = best_legit_v9(v9, "level", point_stat)
            rnd_best = best_legit_v9(v9, "random", point_stat)
            v6_best = best_legit_v6(v6_index, point_stat)
            if None in (lvl0, rnd0, lvl_best, rnd_best, v6_best):
                continue
            rows.append(
                {
                    "decoding": point_stat,
                    "split": SPLIT_SHORT[split],
                    "lvl0": rmse_of(lvl0, split),
                    "rnd0": rmse_of(rnd0, split),
                    "lvl_icl": rmse_of(lvl_best[1], split),
                    "rnd_icl": rmse_of(rnd_best[1], split),
                    "v6_icl": rmse_of(v6_best[1], split),
                }
            )
    if rows:
        lines.append(
            "| decoding | split | ICF-level k0 | ICF-random k0 | ICF-level+ICL | "
            "ICF-random+ICL | v6 ft+ICL | level−random (ICL) | level−v6 (ICL) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['decoding']} | {r['split']} | {r['lvl0']:.2f} | {r['rnd0']:.2f} | "
                f"{r['lvl_icl']:.2f} | {r['rnd_icl']:.2f} | {r['v6_icl']:.2f} | "
                f"{r['lvl_icl'] - r['rnd_icl']:+.2f} | {r['lvl_icl'] - r['v6_icl']:+.2f} |"
            )
    lines.append(VERDICT)
    return lines


#: Finalized after the eval run (data-driven scaffold above stays). Numbers are
#: mean decoding unless noted; paired bootstrap over traces, n=6 ID / 5 OOD.
VERDICT = """
**Verdict — ICF teaches the model to USE level-matched demonstrations (the
oracle proves it), but does NOT beat the base-pretrained ICL ability under
realistic retrieval; retrieval quality stays the binding constraint.**

(1) **ICF makes the model demonstration-DEPENDENT.** Both ICF checkpoints
collapse at k=0 (ICF-level 59.76 ID, ICF-random 46.52, vs the single-trace v6
ft model's 22.20) — trained always with k∈{1,3,5} demos, they are poor without
any. ICL then helps both *massively and significantly* (ctx_level k=5 vs own
k0: level −29.7 ID p=0.012, random −29.0 ID p=0.000). So "did it learn to use
demonstrations?" is unambiguously YES — the model became demo-driven where the
single-trace v6 model was not.

(2) **The random control proves the usage is LEVEL-SPECIFIC — the oracle is the
smoking gun.** Given perfectly level-matched demos (the cheating oracle),
ICF-level dominates ICF-random: **OOD 7.49 vs 61.82 (Δ−54.3, CI [−85.6,
−11.6], p_boot 0.000)** and ID 22.20 vs 32.80 (Δ−10.6, p_boot 0.042). The level
model's oracle gain over its own k0 is large and significant (ID −37.6, OOD
−27.6); the random model gains essentially nothing from oracle demos (R+ICL vs
R k0 at oracle: ID −13.7 p=0.164, OOD −2.9 p=0.673) and oracle demos even make
it WORSE than its own random-demo cell (32.80 vs 20.43 ID). Trained on
mixed-level demos, the random control learned to *ignore* demo level, so it
cannot exploit level-matched ones; trained on level-matched demos, the level
model learned exactly that. This is the cleanest separation in the study and it
is what the control was designed to detect.

(3) **But realistic retrieval cannot deliver oracle-quality matches, so
demonstration-training does NOT beat inherited ICL.** Under `ctx_level`
retrieval ICF-level ≈ the v6 single-trace ft model (k=10: ID −0.27 n.s., OOD
+1.9 n.s.; k=5: ID −3.5 n.s., OOD +2.4 n.s.) and the project's best legitimate
ID (15.63, v6 ft mmr_euclid @ the 512 window) is unbeaten. The ICF advantage
materializes only at the oracle — the same Phase-7 wall: the pool holds a level
twin for every query but no 80-step-context distance reliably finds it. ICF
moved the *ceiling* (level-oracle OOD 7.49 beats v6's oracle 10.89, p_boot
0.000), not the *realized* number.

(4) **A sharp ID/OOD personality split between the two ICF models under
realistic retrieval.** ICF-random is ID-optimised — ctx_level k=10 reaches
16.35 ID (its best, and close to the project floor) but catastrophic 57.2 OOD;
it learned a level-blind demo-amplitude trick that helps the tightly-clustered
ID levels and blows up on the wide OOD levels (and the oracle cannot rescue it).
ICF-level is balanced (ctx_level ≈ 30 ID / 30 OOD) because it relies on level
matching, which transfers to OOD. Neither dominates: random wins ID, level wins
OOD, mirroring the model-free / v6 "shape for ID, level for OOD" split one layer
up — here it is the TRAINING-demo distribution, not the eval retriever, that
sets which axis the model optimises.

**Bottom line.** Demonstration-training works as intended — the model learns to
use (level-matched) demonstrations, definitively shown by the oracle control —
but it does not lift the realized benchmark past the inherited ICL ability,
because inference-time retrieval over an 80-step context still cannot supply the
level-matched demonstrations the trained model now knows how to exploit. The
result is a strong joint statement: *the bottleneck was never the model's
in-context capacity (pretrained or ICF-trained) — it is retrieval, and closing
it needs side information, not more in-context training.* Caveats: n=6 ID / 5
OOD ⇒ most head-to-head differences are not individually significant (the OOD
oracle separation, p_boot 0.000, is the robust headline); two single training
runs; ICF models at the 2048 window vs v6 at 8192 (each at its own training
window); the random control's strong ID is real but level-blind (the oracle
exposes it).
""".rstrip()


########################################################
# Self-test (synthetic results, no files)
########################################################


def _synth_result(method: str, slug: str, strategy: str, k: int, point_stat: str,
                   window: int | None, checkpoint: str | None, id_err: float,
                   ood_err: float, seeds=(42,)) -> dict:
    """Minimal FewShotRunResults-shaped dict for the self-test."""
    id_keys = [f"id_{i}" for i in range(6)]
    ood_keys = [f"ood_{i}" for i in range(5)]
    per_seed = []
    for seed in seeds:
        per_trace = []
        for key in id_keys:
            per_trace.append({"trace_key": key, "true_tail_mean": 100.0,
                              "pred_tail_mean": 100.0 + id_err, "error": id_err,
                              "abs_error": abs(id_err), "squared_error": id_err ** 2})
        for key in ood_keys:
            per_trace.append({"trace_key": "ood_" + key, "true_tail_mean": 100.0,
                              "pred_tail_mean": 100.0 + ood_err, "error": ood_err,
                              "abs_error": abs(ood_err), "squared_error": ood_err ** 2})
        per_seed.append({"seed": seed, "example_ids": {}, "per_trace": per_trace,
                         "in_distribution": {"rmse": abs(id_err), "se_rmse": 0.0},
                         "out_of_distribution": {"rmse": abs(ood_err), "se_rmse": 0.0}})
    config = {"model_slug": slug, "k_shot": k, "point_stat": point_stat,
              "normalization": "shared", "model_context_window": window,
              "checkpoint": checkpoint}
    return {"timestamp": "20260613_000000", "method": method, "config": config,
            "seeds": list(seeds), "n_seeds": len(seeds),
            "in_distribution": {"rmse": abs(id_err), "se_rmse": 0.0, "n_samples": 6,
                                "rmse_std_seeds": None},
            "out_of_distribution": {"rmse": abs(ood_err), "se_rmse": 0.0, "n_samples": 5,
                                    "rmse_std_seeds": None},
            "per_seed": per_seed}


def self_test() -> None:
    print("analyze_icl_finetuned self-test (synthetic results)...")

    def v9_method(mode, strategy, point_stat):
        slug = ICF_SLUGS[mode]
        variant = "shared" + ("-mean" if point_stat == "mean" else "") + f"-win{ICF_EVAL_WINDOW}"
        return f"{slug.replace('/', '_')}_{strategy}__{variant}"

    v9_results = []
    # ICF-level: strong ICL gain; ICF-random: flat. v6: middling.
    for mode, k0_err, icl_err in (("level", 22.0, 16.0), ("random", 24.0, 23.0)):
        ck = f"lora_weights.pt@{mode}00000000"
        for point_stat in ("mean", "median"):
            v9_results.append(_synth_result(
                v9_method(mode, "zeroshot", point_stat), ICF_SLUGS[mode], "zeroshot", 0,
                point_stat, ICF_EVAL_WINDOW, ck, k0_err, k0_err + 10))
            for strategy, k in ICL_CONFIGS:
                err = icl_err if strategy != "oracle_tail" else icl_err - 8
                v9_results.append(_synth_result(
                    v9_method(mode, strategy, point_stat), ICF_SLUGS[mode], strategy, k,
                    point_stat, ICF_EVAL_WINDOW, ck, err, err + 10))
            v9_results.append(_synth_result(
                v9_method(mode, "random", point_stat), ICF_SLUGS[mode], "random", RANDOM_K,
                point_stat, ICF_EVAL_WINDOW, ck, k0_err + 5, k0_err + 12, seeds=(0, 1, 2)))

    v9, checkpoints = build_v9_index(v9_results)
    assert set(checkpoints) == {"level", "random"}, checkpoints
    assert v9_get(v9, "level", "zeroshot", 0, "mean")["config"]["k_shot"] == 0
    # window assert fires on a wrong window
    bad = _synth_result(v9_method("level", "ctx_level", "mean"), ICF_SLUGS["level"],
                        "ctx_level", 5, "mean", 512, "x@y", 1.0, 1.0)
    try:
        build_v9_index([bad])
        raise AssertionError("window assert should fire")
    except AssertionError as e:
        assert "window" in str(e)
    print("✓ build_v9_index: two modes, checkpoint per mode, window asserted")

    bl = best_legit_v9(v9, "level", "mean")
    assert bl is not None and bl[0][0] != "oracle_tail", "best_legit must exclude oracle"
    assert abs(rmse_of(bl[1], "in_distribution") - 16.0) < 1e-9
    print(f"✓ best_legit_v9: ICF-level best legit {bl[0][0]} k={bl[0][1]} ID 16.0 (oracle excluded)")

    # v6 index from synthetic v6 ft cells (full window, FINETUNED_SLUG)
    def v6_method(strategy, point_stat):
        variant = "shared" + ("-mean" if point_stat == "mean" else "")
        return f"{FINETUNED_SLUG.replace('/', '_')}_{strategy}__{variant}"

    v6_results = []
    for point_stat in ("mean", "median"):
        v6_results.append(_synth_result(v6_method("zeroshot", point_stat), FINETUNED_SLUG,
                                        "zeroshot", 0, point_stat, None, "ck@v6", 22.0, 34.0))
        for strategy, k in ICL_CONFIGS:
            v6_results.append(_synth_result(v6_method(strategy, point_stat), FINETUNED_SLUG,
                                            strategy, k, point_stat, None, "ck@v6", 18.6, 36.0))
    v6_index, _ = build_v6_index(v6_results)
    b6 = best_legit_v6(v6_index, "mean")
    assert b6 is not None and abs(rmse_of(b6[1], "in_distribution") - 18.6) < 1e-9
    print(f"✓ best_legit_v6: v6 ft best legit {b6[0][0]} k={b6[0][1]} ID 18.6")

    # report blocks render without error and contain the headline rows
    headline = make_headline(v9, v6_index, checkpoints)
    assert any("ICF-level" in ln for ln in headline)
    assert any("v6 single-trace ft" in ln for ln in headline)
    full = make_full_table(v9, v6_index)
    assert any("Δ level−random" in ln for ln in full)
    paired = make_paired_block(v9, v6_index)
    assert any("L vs R" in ln for ln in paired)
    assert any("L vs v6 ft" in ln for ln in paired)
    ladder = make_ladder_block(v9, v6_index)
    assert any("ICF-level + ICL" in ln for ln in ladder)
    verdict = make_verdict(v9, v6_index)
    assert any("level−random" in ln for ln in verdict)
    print("✓ report blocks render (headline, full table, paired, ladder, verdict)")

    print("\n✅ analyze_icl_finetuned self-test passed!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Phase-9 ICF eval grid")
    parser.add_argument("--v9-dir", type=Path, default=DEFAULT_V9_DIR)
    parser.add_argument("--v6-dir", type=Path, default=DEFAULT_V6_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    v9, checkpoints = build_v9_index(load_results(args.v9_dir))
    assert v9, f"No ICF results in {args.v9_dir}"
    v6_index, _ = build_v6_index(load_results(args.v6_dir))
    print(f"✓ loaded ICF modes {sorted(checkpoints)}; v6 ft baseline from {args.v6_dir.name}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = make_headline(v9, v6_index, checkpoints)
    kcurve_path = fig_kcurve(v9, v6_index, args.out_dir)
    lines.append(f"\n![{kcurve_path.stem}]({kcurve_path.name})")
    lines.extend(make_full_table(v9, v6_index))
    lines.extend(make_paired_block(v9, v6_index))
    lines.extend(make_ladder_block(v9, v6_index))
    lines.extend(make_verdict(v9, v6_index))

    markdown = "\n".join(lines) + "\n"
    table_path = args.out_dir / "icl_finetuning_table.md"
    table_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {table_path}")
    print(f"Wrote {kcurve_path}")


if __name__ == "__main__":
    main()
