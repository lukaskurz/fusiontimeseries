"""Analyze the train-side decoding calibration: q*, k-fold stability, transfer.

Consumes the per-(model, evalset, config, spec, trace) rows written by
``calibrate_decoding.py`` and answers the three questions the 2026-06-14 sweep
could not, because it only ever saw the 11 benchmark traces:

1. **What does each model's decoding curve look like on 244 held-out training
   traces?** ``q*_cal`` = the argmin there, selected without touching the test
   labels.
2. **Is that argmin a stable statistic or the noise of one draw?** 5-fold CV
   stratified by tail level: each fold selects its own q on 4/5 of the
   calibration traces and is scored on the 1/5 it never saw, so the reported
   held-out RMSE prices in the cost of the selection itself.
3. **Does it transfer?** Apply ``q*_cal`` to the benchmark and compare against
   the shipped ``mean``/``median`` decoding and against ``q*_test`` — the
   test-selected quantile, which is the 06-14 number and an upper bound no
   deployable rule can reach.

Bootstrap CIs on the benchmark deltas are paired over traces (n=6 ID / 5 OOD);
with that n they are wide by construction and are reported as such.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_decoding_calibration \
        [--jsonl results/few_shot_v8_calibration/decoding_calibration.jsonl]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_JSONL = Path("results/few_shot_v8_calibration/decoding_calibration.jsonl")
N_FOLDS = 5
N_BOOTSTRAP = 10_000
#: Print order; models absent from the JSONL are skipped.
MODEL_ORDER = ("chronos2", "chronos_bolt", "tirex", "timesfm", "ft_chronos2")
CONFIG_ORDER = (("zeroshot", 0), ("mmr_euclid", 5))


def rmse(rows: list[dict]) -> float:
    t = np.array([r["true_tail"] for r in rows])
    p = np.array([r["pred_tail"] for r in rows])
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mean_bias(rows: list[dict]) -> float:
    """Mean signed tail error (pred - true). >0 = over-prediction."""
    return float(np.mean([r["pred_tail"] - r["true_tail"] for r in rows]))


def over_rate(rows: list[dict]) -> float:
    """Fraction of traces whose predicted tail exceeds the true tail."""
    return float(np.mean([r["pred_tail"] > r["true_tail"] for r in rows]))


def spec_sort_key(spec: str) -> tuple[int, float]:
    """Quantiles ascending, then mean, then meanhead."""
    if spec.startswith("q"):
        return (0, float(spec[1:]))
    return (1, 0.0 if spec == "mean" else 1.0)


def bootstrap_delta(a: list[dict], b: list[dict], seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap on RMSE(a) - RMSE(b) over traces. Returns (diff, lo, hi)."""
    by_trace_a = {r["trace"]: r for r in a}
    by_trace_b = {r["trace"]: r for r in b}
    keys = sorted(set(by_trace_a) & set(by_trace_b))
    sq_a = np.array([(by_trace_a[k]["pred_tail"] - by_trace_a[k]["true_tail"]) ** 2 for k in keys])
    sq_b = np.array([(by_trace_b[k]["pred_tail"] - by_trace_b[k]["true_tail"]) ** 2 for k in keys])
    diff = float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(N_BOOTSTRAP, len(keys)))
    boot = np.sqrt(sq_a[idx].mean(axis=1)) - np.sqrt(sq_b[idx].mean(axis=1))
    return diff, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def stratified_folds(traces: list[str], levels: dict[str, float], n_folds: int) -> list[list[str]]:
    """Fold assignment by rank on tail level (serpentine over the sorted order)."""
    order = sorted(traces, key=lambda t: levels[t])
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, trace in enumerate(order):
        folds[i % n_folds].append(trace)
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the decoding calibration sweep")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--out", type=Path, default=None, help="write the report as markdown")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.open() if line.strip()]

    # Drop incomplete (model, strategy, k, spec) groups — a sweep still running
    # leaves partial specs behind, and a spec scored on a subset of traces is not
    # comparable to one scored on all of them.
    n_queries = len({(r["evalset"], r["trace"]) for r in rows})
    group_counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        group_counts[(r["model"], r["strategy"], r["k"], r["spec"])] += 1
    complete = {g for g, n in group_counts.items() if n == n_queries}
    dropped = sorted(set(group_counts) - complete)
    rows = [r for r in rows if (r["model"], r["strategy"], r["k"], r["spec"]) in complete]

    # A config is analyzable only once its LAST spec is on disk. calibrate_decoding
    # writes the quantile levels first, then "mean", then (TimesFM) "meanhead" — so
    # the terminal spec's presence certifies the whole spec set. Without this a
    # still-running model contributes a truncated grid and a bogus argmin.
    have: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        have[(r["model"], r["strategy"], r["k"])].add(r["spec"])
    def _finished(cfg: tuple) -> bool:
        terminal = "meanhead" if cfg[0] == "timesfm" else "mean"
        return terminal in have[cfg]
    partial = [cfg for cfg in have if not _finished(cfg)]
    rows = [r for r in rows if _finished((r["model"], r["strategy"], r["k"]))]
    if partial:
        print(f"[skipping {len(partial)} unfinished configs: "
              f"{', '.join('/'.join(map(str, c)) for c in sorted(partial))}]\n")
    if dropped:
        print(f"[skipping {len(dropped)} incomplete groups: "
              f"{', '.join('/'.join(map(str, g)) for g in dropped[:6])}"
              f"{' ...' if len(dropped) > 6 else ''}]\n")
    # (model, strategy, k, spec, evalset, split) -> rows
    idx: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        idx[(r["model"], r["strategy"], r["k"], r["spec"], r["evalset"], r["split"])].append(r)

    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Decoding calibration on training traces\n")
    n_cal = len({r["trace"] for r in rows if r["evalset"] == "calibration"})
    emit(f"Calibration set: {n_cal} non-benchmark pool traces, leave-one-out retrieval, "
         f"same 80-step context / tail-80 protocol as the benchmark.\n")

    summary: list[tuple] = []

    for model in models:
        emit(f"\n## {model}\n")
        for strategy, k in CONFIG_ORDER:
            specs = sorted(
                {r["spec"] for r in rows
                 if r["model"] == model and r["strategy"] == strategy and r["k"] == k},
                key=spec_sort_key,
            )
            if not specs:
                continue
            cal_rows = {s: idx[(model, strategy, k, s, "calibration", "cal")] for s in specs}
            if not all(cal_rows.values()):
                continue
            id_rows = {s: idx[(model, strategy, k, s, "benchmark", "id")] for s in specs}
            ood_rows = {s: idx[(model, strategy, k, s, "benchmark", "ood")] for s in specs}

            cal_rmse = {s: rmse(cal_rows[s]) for s in specs}
            id_rmse = {s: rmse(id_rows[s]) for s in specs}
            ood_rmse = {s: rmse(ood_rows[s]) for s in specs}

            cal_bias = {s: mean_bias(cal_rows[s]) for s in specs}
            cal_over = {s: over_rate(cal_rows[s]) for s in specs}

            # Three calibration estimators, all blind to the test labels:
            #   q_cal  — argmin calibration RMSE (the 06-14 rule, honest split)
            #   q_bias — zero the mean signed tail error (bias correction proper)
            #   q_cov  — 50% empirical coverage of the true tail (conformal-flavored)
            q_cal = min(specs, key=lambda s: cal_rmse[s])
            q_bias = min(specs, key=lambda s: abs(cal_bias[s]))
            q_cov = min(specs, key=lambda s: abs(cal_over[s] - 0.5))
            q_test = min(specs, key=lambda s: id_rmse[s])

            emit(f"### {strategy} k={k}\n")
            emit("| spec | calib RMSE (n=%d) | calib bias | calib P(pred>true) | "
                 "bench ID (n=6) | bench OOD (n=5) |" % n_cal)
            emit("|---|---|---|---|---|---|")
            for s in specs:
                mark = []
                if s == q_cal:
                    mark.append("**argmin**")
                if s == q_bias:
                    mark.append("**bias0**")
                if s == q_cov:
                    mark.append("**cov50**")
                if s == q_test:
                    mark.append("_test-argmin_")
                emit(f"| {s} {' '.join(mark)} | {cal_rmse[s]:.2f} | {cal_bias[s]:+.2f} | "
                     f"{cal_over[s]:.2f} | {id_rmse[s]:.2f} | {ood_rmse[s]:.2f} |")
            emit("")

            # --- k-fold: select on 4/5, score on the held-out 1/5 ---
            levels = {r["trace"]: r["true_tail"] for r in cal_rows[specs[0]]}
            traces = sorted(levels)
            folds = stratified_folds(traces, levels, N_FOLDS)
            by_spec_trace = {
                s: {r["trace"]: r for r in cal_rows[s]} for s in specs
            }
            fold_picks: list[str] = []
            held_out: dict[str, list[dict]] = {"median": [], "mean": [], "cv": []}
            for fold in folds:
                held = set(fold)
                in_fold = [t for t in traces if t not in held]
                pick = min(specs, key=lambda s: rmse([by_spec_trace[s][t] for t in in_fold]))
                fold_picks.append(pick)
                held_out["cv"] += [by_spec_trace[pick][t] for t in fold]
                held_out["median"] += [by_spec_trace["q0.50"][t] for t in fold]
                if "mean" in by_spec_trace:
                    held_out["mean"] += [by_spec_trace["mean"][t] for t in fold]
            picks_str = ", ".join(fold_picks)
            agree = len(set(fold_picks)) == 1
            emit(f"**{N_FOLDS}-fold selection** (stratified by tail level) picks: {picks_str} "
                 f"→ {'unanimous' if agree else 'SPLIT'}")
            emit(f"Held-out calibration RMSE: median {rmse(held_out['median']):.2f} | "
                 f"mean {rmse(held_out['mean']):.2f} | CV-selected q {rmse(held_out['cv']):.2f}\n")

            # --- transfer to the benchmark ---
            for split, brows, brmse in (("ID", id_rows, id_rmse), ("OOD", ood_rows, ood_rmse)):
                emit(f"**Transfer {split}** (baselines: median {brmse['q0.50']:.2f}, "
                     f"mean {brmse['mean']:.2f}; test-argmin {q_test} → {brmse[q_test]:.2f} "
                     f"= unreachable upper bound)")
                for name, q_sel in (("argmin", q_cal), ("bias0", q_bias), ("cov50", q_cov)):
                    d_med, lo_m, hi_m = bootstrap_delta(brows[q_sel], brows["q0.50"])
                    d_mean, lo_a, hi_a = bootstrap_delta(brows[q_sel], brows["mean"])
                    emit(f"  - {name:6} → {q_sel} → {brmse[q_sel]:6.2f} | "
                         f"vs median {d_med:+6.2f} [{lo_m:+.2f}, {hi_m:+.2f}] | "
                         f"vs mean {d_mean:+6.2f} [{lo_a:+.2f}, {hi_a:+.2f}]")
            emit("")

            # --- ft contamination split ---
            clean = [r for r in cal_rows[specs[0]] if not r["in_ft_train"]]
            if model.startswith("ft_") and clean:
                clean_keys = {r["trace"] for r in clean}
                clean_rmse = {
                    s: rmse([r for r in cal_rows[s] if r["trace"] in clean_keys]) for s in specs
                }
                q_clean = min(specs, key=lambda s: clean_rmse[s])
                emit(f"**Contamination check**: {len(clean_keys)} of {n_cal} calibration traces "
                     f"are outside `TRAIN_IDXS` (the finetuning val set). "
                     f"argmin on those only: {q_clean} ({clean_rmse[q_clean]:.2f}); "
                     f"argmin on all {n_cal}: {q_cal} ({cal_rmse[q_cal]:.2f}).\n")

            summary.append((model, f"{strategy} k={k}", q_cal, q_bias, q_cov, q_test,
                            sorted(set(fold_picks)), id_rmse["q0.50"], id_rmse["mean"],
                            id_rmse[q_cal], id_rmse[q_bias], id_rmse[q_test],
                            cal_bias["q0.50"], cal_rmse["q0.50"]))

    # ------------------------------------------------------------------
    # Level-conditional optimum: is "decode higher" global, or a function of
    # the query's saturation level? The calibration set is large enough to
    # stratify; the 6 ID / 5 OOD benchmark traces never were.
    # ------------------------------------------------------------------
    emit("\n## Level-conditional optimum\n")
    cal_levels = {r["trace"]: r["true_tail"] for r in rows if r["evalset"] == "calibration"}
    pool_levels = np.array(sorted(cal_levels.values()))
    emit(f"Calibration tail levels (n={len(pool_levels)}): "
         f"p10 {np.percentile(pool_levels, 10):.1f} | p25 {np.percentile(pool_levels, 25):.1f} | "
         f"median {np.median(pool_levels):.1f} | p75 {np.percentile(pool_levels, 75):.1f} | "
         f"p90 {np.percentile(pool_levels, 90):.1f}\n")
    for split in ("id", "ood"):
        lv = sorted({r["trace"]: r["true_tail"] for r in rows
                     if r["evalset"] == "benchmark" and r["split"] == split}.values())
        pct = [float((pool_levels < x).mean() * 100) for x in lv]
        emit(f"- benchmark {split.upper()} levels {[f'{x:.1f}' for x in lv]} "
             f"→ pool percentiles {[f'{q:.0f}%' for q in pct]}")
    emit("")

    ordered_traces = sorted(cal_levels, key=lambda t: cal_levels[t])
    n_bins = 4
    bins = [ordered_traces[i * len(ordered_traces) // n_bins:
                           (i + 1) * len(ordered_traces) // n_bins] for i in range(n_bins)]

    emit("\nArgmin quantile per calibration level quartile — plus the ceiling a "
         "*level-aware* picker would reach if it knew which quartile the query is in "
         "(`oracle-by-level`), against the best single global quantile.\n")
    emit("| model | config | " + " | ".join(f"Q{i + 1}" for i in range(n_bins))
         + " | best global | global RMSE | oracle-by-level RMSE | gap |")
    emit("|---|---|" + "---|" * (n_bins + 4))
    for model in models:
        for strategy, k in CONFIG_ORDER:
            specs = sorted(
                {r["spec"] for r in rows if r["model"] == model
                 and r["strategy"] == strategy and r["k"] == k},
                key=spec_sort_key,
            )
            if not specs:
                continue
            by_spec = {s: {r["trace"]: r
                           for r in idx[(model, strategy, k, s, "calibration", "cal")]}
                       for s in specs}
            if not all(by_spec.values()):
                continue
            picks, oracle_rows = [], []
            for b in bins:
                best = min(specs, key=lambda s: rmse([by_spec[s][t] for t in b]))
                picks.append(best)
                oracle_rows += [by_spec[best][t] for t in b]
            g_best = min(specs, key=lambda s: rmse(list(by_spec[s].values())))
            g_rmse = rmse(list(by_spec[g_best].values()))
            o_rmse = rmse(oracle_rows)
            emit(f"| {model} | {strategy} k={k} | " + " | ".join(picks)
                 + f" | {g_best} | {g_rmse:.2f} | {o_rmse:.2f} | {g_rmse - o_rmse:+.2f} |")
    emit("")

    emit("\n## Summary — calibration-selected decoding vs. the shipped knob\n")
    emit("| model | config | argmin | bias0 | cov50 | fold picks | q*_test | "
         "ID median | ID mean | ID @argmin | ID @bias0 | ID @q*_test |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (m, c, qc, qb, qv, qt, picks, r_med, r_mean, r_qc, r_qb, r_qt, _, _) in summary:
        emit(f"| {m} | {c} | {qc} | {qb} | {qv} | {'/'.join(picks)} | {qt} | "
             f"{r_med:.2f} | {r_mean:.2f} | **{r_qc:.2f}** | {r_qb:.2f} | _{r_qt:.2f}_ |")

    emit("\n## Mechanism: does the optimal level track the residual under-prediction?\n")
    emit("The 06-14 claim is that \"decode higher\" is a global level-bias correction whose "
         "size tracks how badly the config under-predicts. On the calibration set that is "
         "directly measurable: median-decoding bias vs. the selected quantile.\n")
    emit("| model | config | median-decode calib bias | calib RMSE @median | q*_cal |")
    emit("|---|---|---|---|---|")
    pts = []
    for (m, c, qc, _qb, _qv, _qt, _p, _rm, _ra, _rc, _rb, _rt, bias, rmed) in summary:
        emit(f"| {m} | {c} | {bias:+.2f} | {rmed:.2f} | {qc} |")
        if qc.startswith("q"):
            pts.append((bias, float(qc[1:])))
    if len(pts) >= 3:
        b = np.array([p[0] for p in pts]); q = np.array([p[1] for p in pts])
        emit(f"\nPearson r(median-decode bias, q*_cal) = {np.corrcoef(b, q)[0, 1]:+.3f} "
             f"over {len(pts)} (model, config) cells "
             f"— negative = more under-prediction wants a higher quantile.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\n✅ wrote {args.out}")


if __name__ == "__main__":
    main()
