"""Analysis of the Phase-5 decoding grid: tables, figure, ensembles.

Loads ``results/few_shot_v5_decoding/`` and produces, in
``docs/results/fewshot/``:

- ``decoding_table.md`` — per-model median/mean(/meanhead) tables for the
  four configs (anchor / best / oracle / random), paired comparisons mean
  vs median per (model, config), the seed-ensembling block, the
  cross-model-ensemble block, the decision paragraph, and interpretation
  notes;
- ``decoding_effect.png`` — per model x config, paired median→mean
  (→meanhead) dots/arrows, two panels ID/OOD.

Method labels are parsed as ``{slug with / -> _}_{strategy}__{variant}``
with the v5 token set (``shared``; ``mean`` | ``meanhead``) and
cross-checked against the FewShotConfig metadata. The analyzer hard-asserts
per-seed example-id equality for every mean/meanhead cell against its
median twin (same select_fn + same seeds by construction).

Ensembling is pure post-processing: the tail mean is linear, so averaging
forecasts before scoring ≡ averaging the recorded ``pred_tail_mean`` values
— across seeds (seed ensembling, from the 20-seed random cells) or across
models (cross-model ensembling, from the deterministic best-config cells).
Ensembled runs are synthesized as single-seed pseudo-results dicts so the
existing ``paired_comparison`` applies unchanged.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_decoding
"""

import argparse
import itertools
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
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import MODEL_SLUGS
from fusiontimeseries.benchmarking.few_shot.run_decoding_grid import (
    BEST_CONFIGS,
    ORACLE_K,
    RANDOM_K,
)
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    rmse_with_standard_error,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_DIR: Path = REPO_ROOT / "results" / "few_shot_v5_decoding"
DEFAULT_OUT_DIR: Path = REPO_ROOT / "docs" / "results" / "fewshot"

SPLITS: tuple[str, ...] = ("in_distribution", "out_of_distribution")
SPLIT_SHORT: dict[str, str] = {"in_distribution": "ID", "out_of_distribution": "OOD"}
MODEL_ORDER: tuple[str, ...] = ("tirex", "timesfm", "chronos2", "chronos_bolt")
POINT_STATS: tuple[str, ...] = ("median", "mean", "meanhead")

#: Index key: (slug, strategy, k, point_stat)
Key = tuple[str, str, int, str]

#: The four grid configs as (label, strategy_for(model), k_for(model)).
CONFIG_ROWS: list[tuple[str, str | None, int | None]] = [
    ("zero-shot k=0", "zeroshot", 0),
    ("best", None, None),  # per-model BEST_CONFIGS
    (f"oracle_tail k={ORACLE_K} (cheats)", "oracle_tail", ORACLE_K),
    (f"random k={RANDOM_K} (20 seeds)", "random", RANDOM_K),
]

MODEL_STYLE: dict[str, dict] = {
    "tirex": {"color": "tab:green"},
    "timesfm": {"color": "tab:purple"},
    "chronos2": {"color": "tab:blue"},
    "chronos_bolt": {"color": "tab:orange"},
}


def config_for(model_name: str, row: tuple[str, str | None, int | None]) -> tuple[str, str, int]:
    """Resolve a CONFIG_ROWS entry to (label, strategy, k) for one model."""
    label, strategy, k = row
    if strategy is None:
        strategy, k = BEST_CONFIGS[model_name]
        label = f"best: {strategy} k={k}"
    return label, strategy, k


########################################################
# Parsing and indexing
########################################################


def parse_variant(variant: str) -> dict | None:
    """Decode a v5 hyphen-joined variant label into config fields."""
    fields: dict = {"normalization": "per_example", "point_stat": "median"}
    if variant == "base":
        return fields
    for token in variant.split("-"):
        if token == "shared":
            fields["normalization"] = "shared"
        elif token in ("mean", "meanhead"):
            fields["point_stat"] = token
        else:
            return None
    return fields


def build_index(results: list[dict]) -> dict[Key, dict]:
    """Index v5 results; latest timestamp wins; tokens cross-checked."""
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
        key: Key = (slug, strategy, int(config["k_shot"]), fields["point_stat"])
        if key not in index or r["timestamp"] > index[key]["timestamp"]:
            index[key] = r
    return index


def get(
    index: dict[Key, dict], model_name: str, strategy: str, k: int, point_stat: str
) -> dict | None:
    return index.get((MODEL_SLUGS[model_name], strategy, k, point_stat))


def rmse_of(result: dict, split: str) -> float:
    return float(result[split]["rmse"])


def std_of(result: dict, split: str) -> float | None:
    value = result[split].get("rmse_std_seeds")
    return None if value is None else float(value)


def assert_decoding_twin_example_ids(index: dict[Key, dict]) -> int:
    """Hard-assert per-seed example-id equality vs the median twin."""
    checked = 0
    for key, alt_result in index.items():
        slug, strategy, k, point_stat = key
        if point_stat == "median" or strategy == "zeroshot":
            continue
        twin = index.get((slug, strategy, k, "median"))
        assert twin is not None, f"Missing median twin for {key}"
        assert alt_result["seeds"] == twin["seeds"], f"Seed mismatch for {key}"
        for sa, st in zip(alt_result["per_seed"], twin["per_seed"]):
            assert sa["example_ids"] == st["example_ids"], (
                f"example_ids differ for {key} seed {sa['seed']}"
            )
        checked += 1
    return checked


########################################################
# Ensembling (post-processing over recorded tail means)
########################################################


def _split_summary(per_trace: list[dict], split: str) -> dict:
    is_ood = split == "out_of_distribution"
    records = [tr for tr in per_trace if tr["trace_key"].startswith("ood_") == is_ood]
    rmse, se = rmse_with_standard_error(
        np.array([tr["true_tail_mean"] for tr in records]),
        np.array([tr["pred_tail_mean"] for tr in records]),
    )
    return {"rmse": float(rmse), "se_rmse": float(se), "n_samples": len(records)}


def synthesize_run(per_trace: list[dict], method: str, config: dict) -> dict:
    """Wrap ensembled per-trace records as a single-seed pseudo-results dict."""
    summaries = {split: _split_summary(per_trace, split) for split in SPLITS}
    return {
        "timestamp": "synthesized",
        "method": method,
        "config": config,
        "seeds": [-1],
        "n_seeds": 1,
        "deterministic": True,
        "in_distribution": summaries["in_distribution"],
        "out_of_distribution": summaries["out_of_distribution"],
        "per_seed": [
            {
                "seed": -1,
                "example_ids": {},
                "in_distribution": summaries["in_distribution"],
                "out_of_distribution": summaries["out_of_distribution"],
                "per_trace": per_trace,
            }
        ],
    }


def _trace_records(result: dict) -> dict[str, dict[int, dict]]:
    """trace_key -> seed -> per-trace record."""
    records: dict[str, dict[int, dict]] = {}
    for seed_result in result["per_seed"]:
        for tr in seed_result["per_trace"]:
            records.setdefault(tr["trace_key"], {})[seed_result["seed"]] = tr
    return records


def _ensembled_per_trace(grouped: dict[str, list[dict]]) -> list[dict]:
    """Average pred_tail_mean per trace over a group of records."""
    per_trace: list[dict] = []
    for trace_key, records in grouped.items():
        true_tail = records[0]["true_tail_mean"]
        assert all(tr["true_tail_mean"] == true_tail for tr in records)
        pred = float(np.mean([tr["pred_tail_mean"] for tr in records]))
        error = pred - true_tail
        per_trace.append(
            {
                "trace_key": trace_key,
                "true_tail_mean": true_tail,
                "pred_tail_mean": pred,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error**2,
            }
        )
    return per_trace


def ensemble_seeds(result: dict) -> dict:
    """Seed ensemble: average pred_tail_mean over all seeds per trace.

    The tail mean is linear in the forecast, so this equals averaging the
    full forecasts across seeds before scoring.
    """
    grouped = {
        trace_key: list(by_seed.values())
        for trace_key, by_seed in _trace_records(result).items()
    }
    return synthesize_run(
        _ensembled_per_trace(grouped),
        method=result["method"] + "+seed_ens",
        config=result["config"],
    )


def ensemble_models(results: list[dict], method: str) -> dict:
    """Cross-model ensemble of deterministic runs (one record per trace)."""
    grouped: dict[str, list[dict]] = {}
    for result in results:
        assert result["n_seeds"] == 1, "cross-model ensembling expects det cells"
        for trace_key, by_seed in _trace_records(result).items():
            grouped.setdefault(trace_key, []).extend(by_seed.values())
    return synthesize_run(
        _ensembled_per_trace(grouped), method=method, config=results[0]["config"]
    )


########################################################
# Tables
########################################################


def fmt_split_pair(result: dict | None) -> str:
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


def make_main_table(index: dict[Key, dict]) -> list[str]:
    lines = ["# Point-forecast decoding and ensembling — Phase 5 results", ""]
    lines.append(
        "All four models, t266 protocol: fixed 245-trace pool, full-length "
        "example targets, context 80, prediction 64, tail 80, flat concat + "
        "SHARED scaling (the Phase-3 winner) everywhere. `median` is the "
        "frozen Phase-1..4 decoding (q0.5 of the 9 deciles); `mean` is the "
        "decile-average estimator (1/9)·Σ q₀.₁..q₀.₉ — biased LOW on "
        "right-skewed data because it truncates the tails beyond q0.1/q0.9; "
        "`meanhead` is TimesFM's native mean output head (index 0 of its "
        "[mean, q0.1..q0.9] forecast — the unbiased-head cross-check; the "
        "other models have no native mean: TiRex's library \"mean\" return "
        "is a relabeled median, confirmed at runtime by smoke D1). Cells are "
        "`ID / OOD` tail RMSE; the 20-seed random cells show mean±std over "
        "seeds. All cells from ONE grid run "
        "(`results/few_shot_v5_decoding/`); identical example sets across "
        "decoding twins (hard-asserted)."
    )
    lines.append("\n## Decoding effect per model and config\n")
    for model_name in MODEL_ORDER:
        lines.append(f"\n### {MODEL_SLUGS[model_name]}\n")
        has_meanhead = model_name == "timesfm"
        header = "| config | median | mean | Δmean−median (ID) | Δ (OOD) |"
        if has_meanhead:
            header += " meanhead | Δmeanhead−median (ID) | Δ (OOD) |"
        lines.append(header)
        lines.append("|" + "---|" * (header.count("|") - 1))
        for row in CONFIG_ROWS:
            label, strategy, k = config_for(model_name, row)
            med = get(index, model_name, strategy, k, "median")
            mean = get(index, model_name, strategy, k, "mean")
            cells = [label, fmt_split_pair(med), fmt_split_pair(mean)]
            for split in SPLITS:
                if med is not None and mean is not None:
                    cells.append(f"{rmse_of(mean, split) - rmse_of(med, split):+.2f}")
                else:
                    cells.append("—")
            if has_meanhead:
                mh = get(index, model_name, strategy, k, "meanhead")
                cells.append(fmt_split_pair(mh))
                for split in SPLITS:
                    if med is not None and mh is not None:
                        cells.append(f"{rmse_of(mh, split) - rmse_of(med, split):+.2f}")
                    else:
                        cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def make_decoding_comparisons(index: dict[Key, dict]) -> list[str]:
    lines = [
        "\n## Paired comparisons — mean vs median per (model, config)",
        "",
        "Δ = RMSE(A) − RMSE(B); negative favours A (mean better). CI/p from "
        "a 10k paired bootstrap over traces; Wilcoxon shown for completeness "
        "(p floors 0.031 ID / 0.0625 OOD). `[trace_seed]` rows appear for "
        "the multi-seed random cells.",
        COMPARISON_HEADER,
    ]
    for model_name in MODEL_ORDER:
        for row in CONFIG_ROWS:
            label, strategy, k = config_for(model_name, row)
            med = get(index, model_name, strategy, k, "median")
            mean = get(index, model_name, strategy, k, "mean")
            if med is None or mean is None:
                continue
            lines.extend(
                comparison_rows(f"{model_name} {label}: mean vs median", mean, med)
            )
    mh_rows: list[str] = []
    for row in CONFIG_ROWS:
        label, strategy, k = config_for("timesfm", row)
        med = get(index, "timesfm", strategy, k, "median")
        mean = get(index, "timesfm", strategy, k, "mean")
        mh = get(index, "timesfm", strategy, k, "meanhead")
        if mh is None:
            continue
        if med is not None:
            mh_rows.extend(
                comparison_rows(f"timesfm {label}: meanhead vs median", mh, med)
            )
        if mean is not None:
            mh_rows.extend(
                comparison_rows(f"timesfm {label}: meanhead vs decile-mean", mh, mean)
            )
    if mh_rows:
        lines.append(
            "\n### TimesFM mean head (native, unbiased) vs decile average"
        )
        lines.append(COMPARISON_HEADER)
        lines.extend(mh_rows)
    return lines


def make_seed_ensemble_block(index: dict[Key, dict]) -> list[str]:
    lines = [
        "\n## Seed ensembling (random k=10, 20 example sets)",
        "",
        "`per-seed` is the standard aggregation (mean over seeds of each "
        "seed's RMSE); `seed-ens` averages the 20 predicted tail means per "
        "trace before scoring (≡ averaging forecasts: the tail mean is "
        "linear). Paired rows test seed-ens vs the per-seed run over traces.",
        "",
        "| model | decoding | per-seed RMSE (ID / OOD) | seed-ens RMSE (ID / OOD) | Δ ID | Δ OOD |",
        "|---|---|---|---|---|---|",
    ]
    comparisons: list[str] = []
    for model_name in MODEL_ORDER:
        for point_stat in POINT_STATS:
            result = get(index, model_name, "random", RANDOM_K, point_stat)
            if result is None or result["n_seeds"] < 2:
                continue
            ens = ensemble_seeds(result)
            deltas = [
                rmse_of(ens, split) - rmse_of(result, split) for split in SPLITS
            ]
            lines.append(
                f"| {model_name} | {point_stat} | {fmt_split_pair(result)} | "
                f"{fmt_split_pair(ens)} | {deltas[0]:+.2f} | {deltas[1]:+.2f} |"
            )
            comparisons.extend(
                comparison_rows(
                    f"{model_name} {point_stat}: seed-ens vs per-seed", ens, result
                )
            )
    lines.append(COMPARISON_HEADER)
    lines.extend(comparisons)
    return lines


def make_model_ensemble_block(index: dict[Key, dict]) -> list[str]:
    lines = [
        "\n## Cross-model ensembling (best configs, deterministic cells)",
        "",
        "Per-trace tail-mean averages across the models' best-config "
        "forecasts (Bolt mmr k=10, TimesFM mmr k=10, TiRex ctx_euclid k=10, "
        "Chronos-2 mmr k=5), per decoding. Compared against the best single "
        "model of that decoding (lowest ID RMSE).",
    ]
    for point_stat in ("median", "mean"):
        cells = {
            model_name: get(index, model_name, *BEST_CONFIGS[model_name], point_stat)
            for model_name in MODEL_ORDER
        }
        cells = {name: r for name, r in cells.items() if r is not None}
        if len(cells) < 2:
            continue
        best_single_name = min(cells, key=lambda n: rmse_of(cells[n], "in_distribution"))
        best_single = cells[best_single_name]
        lines.append(f"\n### Decoding: {point_stat}\n")
        lines.append("| ensemble | ID | OOD | Δ ID vs best single | Δ OOD |")
        lines.append("|---|---|---|---|---|")
        for model_name, result in cells.items():
            marker = " ← best single" if model_name == best_single_name else ""
            lines.append(
                f"| {model_name} (single){marker} | {rmse_of(result, 'in_distribution'):.2f} | "
                f"{rmse_of(result, 'out_of_distribution'):.2f} | — | — |"
            )
        comparisons: list[str] = []
        combos = [
            combo
            for size in range(2, len(cells) + 1)
            for combo in itertools.combinations(sorted(cells), size)
            if size in (2, len(cells))
        ]
        for combo in combos:
            ens = ensemble_models(
                [cells[name] for name in combo], method="+".join(combo)
            )
            deltas = [
                rmse_of(ens, split) - rmse_of(best_single, split) for split in SPLITS
            ]
            lines.append(
                f"| {' + '.join(combo)} | {rmse_of(ens, 'in_distribution'):.2f} | "
                f"{rmse_of(ens, 'out_of_distribution'):.2f} | {deltas[0]:+.2f} | "
                f"{deltas[1]:+.2f} |"
            )
            comparisons.extend(
                comparison_rows(
                    f"[{point_stat}] {'+'.join(combo)} vs {best_single_name}",
                    ens,
                    best_single,
                )
            )
        lines.append(COMPARISON_HEADER)
        lines.extend(comparisons)
    return lines


def make_decision_section(index: dict[Key, dict]) -> list[str]:
    """Data-driven summary feeding the adopt-mean-or-not decision."""
    lines = ["\n## Decision — does mean decoding become the default?", ""]
    improved, regressed, rows = 0, 0, []
    for model_name in MODEL_ORDER:
        for row in CONFIG_ROWS:
            label, strategy, k = config_for(model_name, row)
            med = get(index, model_name, strategy, k, "median")
            mean = get(index, model_name, strategy, k, "mean")
            if med is None or mean is None:
                continue
            delta = rmse_of(mean, "in_distribution") - rmse_of(med, "in_distribution")
            comp = paired_comparison(mean, med, "in_distribution", pairing="trace")
            significant = comp.bootstrap_ci_high < 0 or comp.bootstrap_ci_low > 0
            improved += delta < 0
            regressed += delta > 0
            rows.append((model_name, label, delta, significant))
    lines.append(
        f"Across the {len(rows)} (model, config) cells with both decodings: "
        f"mean improves ID RMSE in {improved}, worsens it in {regressed} "
        "(bootstrap-significant cells marked •):"
    )
    lines.append("")
    lines.append("| model | config | Δ ID (mean−median) | significant |")
    lines.append("|---|---|---|---|")
    for model_name, label, delta, significant in sorted(rows, key=lambda r: r[2]):
        lines.append(
            f"| {model_name} | {label} | {delta:+.2f} | {'•' if significant else ''} |"
        )
    lines.append(DECISION_VERDICT)
    return lines


DECISION_VERDICT = """
**Verdict — adopt mean decoding per model, not globally.** For
**Chronos-2** mean decoding is uniformly better (zero-shot −20.4 ID,
random −5.2 ID significant, oracle −1.7 ID significant, best config
−2.3 ID) — it becomes the DEFAULT for all later Chronos-2 work,
including Phase 5's finetuned-Chronos-2 ICL runs. For **Chronos-Bolt**
and **TiRex** mean is a small free win (ID never worse, the only
regression is TiRex's cheating oracle at +0.35 n.s.); adopt it — it
produces the new best legitimate training-free cell, Bolt mmr_euclid
shared k=10 at **22.63 ID** (was 23.28). For **TimesFM** keep the
median: at its best config both the decile mean (+1.05 ID / +1.57 OOD)
and its own native mean head (+2.01 ID / +3.46 OOD) are
bootstrap-significantly WORSE — and since meanhead ≥ decile-mean ≥
median there, the failure is not the decile truncation bias but the
mean statistic itself interacting badly with TimesFM's wide right tail
under ICL. The pattern across models: the worse the level calibration
of a config, the more the mean helps (anchors ≫ random ≫ best), exactly
what a skew correction should do once shared scaling has already moved
the level most of the way.

**Ensembling.** Seed ensembling (average the 20 random-example-set
forecasts before scoring) is significantly better than per-seed scoring
for every model and decoding (ID −1.7 to −5.7, OOD −1.5 to −4.4, all
bootstrap p ≤ 0.001) — a real variance reduction, but ensembled random
selection (best: Bolt mean 34.27 ID) still loses clearly to plain
retrieval (22.63), so it is a fallback when retrieval is unavailable,
not a replacement. Cross-model ensembling of the best configs never
beats the best single model ID (closest: Bolt+TimesFM −0.31 n.s.; the
TODO's literal Bolt+TiRex is +3.0 WORSE) — the models' tail-level
errors are too correlated for averaging to pay.
""".rstrip()


INTERPRETATION_NOTES = """
## Interpretation notes

- **Why mean decoding at all.** The metric is tail RMSE of a positive,
  right-skewed flux; the RMSE-optimal point forecast is the conditional
  MEAN, and the median of a right-skewed predictive distribution sits
  below it. Phase 3's shared scaling already fixed the *level transfer*
  problem, so the remaining median-vs-mean gap was expected to be small —
  the deliverable is the decision either way.
- **The decile average is not the exact mean.** (1/9)·Σ q₀.₁..q₀.₉
  truncates the predictive distribution beyond q0.1/q0.9, so it
  UNDERestimates the mean of a right-skewed distribution (smoke D3:
  lognormal σ=1 has median 1.00 < decile-avg 1.33 < exact mean 1.65).
  TimesFM's native mean head (`meanhead`) is the unbiased-head
  cross-check for this bias.
- **Decoding feeds back through the rollout.** The decoded point is
  appended to the context and re-normalized each 64-step iteration, so
  mean decoding changes the whole trajectory, not just the final
  read-out; the k=0 anchors isolate this pure decoding+feedback effect.
- **TiRex has NO native mean.** The library's second ``forecast()``
  return is labeled mean but selects q0.5 by index
  (``tirex/models/tirex.py``: ``# median as mean``) — confirmed
  bit-identical to the median at runtime by smoke D1. The TODO's "TiRex
  returns the mean natively" was wrong.
- **TimesFM layout evidence** (smoke D2): the full forecast's last dim is
  ``[mean, q0.1..q0.9]``; the first ``forecast()`` return is literally
  ``full[..., 5]`` (the median; bit-equal), indices 1..9 are monotone,
  and index 0 is a distinct head sitting above the median on a real flux
  trace (+0.17 in normalized space) — consistent with right skew.
- **Ensembling is post-processing.** The recorded per-seed per-trace
  ``pred_tail_mean`` is linear in the forecast, so averaging tail means ≡
  averaging forecasts before scoring; no harness change. Seed ensembling
  removes example-selection variance from the random cells; cross-model
  ensembling averages the deterministic best-config cells.
- MPS is not bit-deterministic across process runs: headline pairs live
  within the single v5 grid run; the v5 median cells double as a
  cross-run reproduction of the v3/v4 twins (reported, not asserted).
- Wilcoxon p floors at 0.031 (n=6 ID) / 0.0625 (n=5 OOD) under
  pairing="trace"; the bootstrap CI is the primary evidence.
""".strip()


########################################################
# Figure
########################################################


def fig_decoding_effect(index: dict[Key, dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    group_width = 1.0
    offsets = np.linspace(-0.3, 0.3, len(MODEL_ORDER))
    for ax, split in zip(axes, SPLITS):
        for gi, row in enumerate(CONFIG_ROWS):
            for model_name, off in zip(MODEL_ORDER, offsets):
                label, strategy, k = config_for(model_name, row)
                med = get(index, model_name, strategy, k, "median")
                mean = get(index, model_name, strategy, k, "mean")
                if med is None or mean is None:
                    continue
                x = gi * group_width + off
                y_med = rmse_of(med, split)
                y_mean = rmse_of(mean, split)
                color = MODEL_STYLE[model_name]["color"]
                ax.annotate(
                    "",
                    xy=(x, y_mean),
                    xytext=(x, y_med),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4},
                )
                ax.plot(x, y_med, "o", mfc="white", mec=color, ms=6, zorder=3)
                ax.plot(x, y_mean, "o", color=color, ms=6, zorder=3)
                mh = get(index, model_name, strategy, k, "meanhead")
                if mh is not None:
                    ax.plot(x, rmse_of(mh, split), "*", color=color, ms=10, zorder=3)
        ax.set_xticks(np.arange(len(CONFIG_ROWS)) * group_width)
        ax.set_xticklabels(
            ["zero-shot\nk=0", "best config", f"oracle_tail\nk={ORACLE_K}", f"random\nk={RANDOM_K}"],
            fontsize=8.5,
        )
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(SPLIT_SHORT[split], fontsize=11)
    axes[0].set_ylabel("tail RMSE")
    model_handles = [
        plt.Line2D([], [], color=MODEL_STYLE[m]["color"], marker="s", ls="", label=m)
        for m in MODEL_ORDER
    ]
    decode_handles = [
        plt.Line2D([], [], marker="o", mfc="white", mec="0.2", color="none", label="median"),
        plt.Line2D([], [], marker="o", color="0.2", ls="", label="mean (decile avg)"),
        plt.Line2D([], [], marker="*", color="0.2", ls="", ms=10, label="meanhead (TimesFM)"),
    ]
    fig.legend(
        handles=model_handles + decode_handles,
        loc="lower center",
        ncol=7,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Point-statistic decoding: median → mean per model and config (shared scaling, v5 grid)",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path = out_dir / "decoding_effect.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Phase-5 decoding grid")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    index = build_index(load_results(args.results_dir))
    assert index, f"No decoding-grid results in {args.results_dir}"
    n_pairs = assert_decoding_twin_example_ids(index)
    print(f"✓ example_ids identical for all {n_pairs} decoding twin pairs")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = make_main_table(index)
    fig_path = fig_decoding_effect(index, args.out_dir)
    lines.append(f"\n![{fig_path.stem}]({fig_path.name})")
    lines.extend(make_decoding_comparisons(index))
    lines.extend(make_seed_ensemble_block(index))
    lines.extend(make_model_ensemble_block(index))
    lines.extend(make_decision_section(index))
    lines.append("\n" + INTERPRETATION_NOTES)

    markdown = "\n".join(lines) + "\n"
    table_path = args.out_dir / "decoding_table.md"
    table_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {table_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
