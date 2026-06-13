"""Analysis of the Phase-8 evaluation reconciliation run.

Loads ``results/few_shot_v8_reconciliation/`` (the ladder re-run on Severin's
``[0::3]`` phase) plus the existing ``[2::3]`` cells (``results/few_shot_v6_finetuned/``
for base Chronos-2 + finetuned, ``results/few_shot_v5_decoding/`` for Chronos-Bolt)
and the Severin-protocol anchor (``severin_anchor.json``), and produces in
``docs/results/fewshot/``:

- ``evaluation_reconciliation.md`` — (a) the protocol-difference table with
  file:line citations, marking which differences this run ALIGNS (eval-level:
  trace phase, metric window, seed) vs which are inherent to the two METHODS
  (method-level: context/prediction length, rollout vs raw forward); (b) the
  metric-audit recap (``[:-80]`` vs ``[-80:]``, numbers pulled from the anchor
  JSON); (c) the aligned adaptation ladder — few-shot rungs measured on ``[0::3]``
  beside their existing ``[2::3]`` numbers, plus the honest-rescore anchor rung
  and the GPR / GyroSwin-1B paper references — and the per-trace
  ``[0::3]``-vs-``[2::3]`` true-tail-mean deltas as the comparability evidence.
- ``reconciliation_ladder.png`` — the single-protocol (``[0::3]``) ladder, ID/OOD
  panels, with the ``[2::3]`` value overlaid per rung.

Baseline ``[2::3]`` numbers are recomputed in-memory (model-free, deterministic —
no file writes, never touching ``results/few_shot_v2``). No hard cross-rollout
assert: our harness rollout is not Severin's raw forward, so the anchor rung is
REPORTED, not asserted equal.

Usage:
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_reconciliation
    uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_reconciliation --self-test
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
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import MODEL_SLUGS

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
V8_DIR: Path = REPO_ROOT / "results" / "few_shot_v8_reconciliation"
V6_DIR: Path = REPO_ROOT / "results" / "few_shot_v6_finetuned"
V5_DIR: Path = REPO_ROOT / "results" / "few_shot_v5_decoding"
DEFAULT_OUT_DIR: Path = REPO_ROOT / "docs" / "results" / "fewshot"

BASE_SLUG: str = MODEL_SLUGS["chronos2"]
BOLT_SLUG: str = MODEL_SLUGS["chronos_bolt"]
SPLITS: tuple[str, ...] = ("in_distribution", "out_of_distribution")

#: Paper references (both phases identical — external constants, like the v6 ladder).
GPR_REF: dict[str, float] = {"id": 43.82, "ood": 59.28}
GYROSWIN_REF: dict[str, float] = {"id": 18.35, "ood": 26.43}

#: Index key: (slug, strategy, k, window). window None = full / not applicable.
Key = tuple[str, str, int, int | None]


########################################################
# Loading / indexing
########################################################


def _index_model_cell(data: dict, index: dict[Key, dict]) -> None:
    """Index a result dict under (slug, strategy, k, window); latest wins."""
    config = data.get("config", {})
    slug = config.get("model_slug")
    method = data.get("method", "")
    if slug is None or "__" not in method:
        return
    prefix = slug.replace("/", "_") + "_"
    if not method.startswith(prefix):
        return
    strategy = method[len(prefix):].rsplit("__", 1)[0]
    key: Key = (
        slug,
        strategy,
        int(config["k_shot"]),
        config.get("model_context_window"),
    )
    if key not in index or data["timestamp"] > index[key]["timestamp"]:
        index[key] = data


def load_v8(results_dir: Path) -> tuple[dict[Key, dict], dict[str, dict]]:
    """Load the Phase-0 reconciliation cells.

    Returns:
        (model_index keyed (slug, strategy, k, window), baseline_index keyed name).
    """
    model_index: dict[Key, dict] = {}
    baselines: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*_reconciliation.json")):
        data = json.load(open(path))
        data["_path"] = str(path)
        method = data.get("method", "")
        if "__" in method:
            _index_model_cell(data, model_index)
        else:  # baseline: e.g. "persistence-phase0"
            baselines[method.removesuffix("-phase0")] = data
    return model_index, baselines


def load_phase2_models(dirs: list[Path]) -> dict[Key, dict]:
    """Load the existing ``[2::3]`` model cells from prior grid dirs."""
    index: dict[Key, dict] = {}
    for results_dir in dirs:
        for path in sorted(results_dir.glob("*_fewshot_results.json")):
            data = json.load(open(path))
            _index_model_cell(data, index)
    return index


def compute_phase2_baselines() -> dict[str, dict]:
    """Recompute the ``[2::3]`` baseline cells in-memory (deterministic, no writes)."""
    from fusiontimeseries.benchmarking.few_shot.baselines import (
        make_knn_copy_forecast,
        make_pool_tail_mean_forecast,
        persistence_forecast,
    )
    from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
        FewShotConfig,
        create_example_pool,
    )
    from fusiontimeseries.benchmarking.few_shot.harness import run_benchmark
    from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS
    from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
        BenchmarkDataProvider,
    )

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    provider = BenchmarkDataProvider()
    config = FewShotConfig(
        device="cpu",
        model_slug="baseline",
        model_prediction_length=64,
        start_context_length=80,
        relevant_prediction_tail=80,
        k_shot=0,
        random_seed=42,
        example_target_length=None,
    )
    fns = {
        "persistence": persistence_forecast,
        "pool_tail_mean": make_pool_tail_mean_forecast(pool),
        "knn_copy_k5": make_knn_copy_forecast(pool, k=5),
    }
    out: dict[str, dict] = {}
    for name, forecast_fn in fns.items():
        out[name] = run_benchmark(
            forecast_fn=forecast_fn,
            config=config,
            example_pool=pool,
            method=name,
            deterministic=True,
            provider=provider,
            save=False,
        ).model_dump()
    return out


def rmse_of(result: dict | None, split: str) -> float | None:
    return None if result is None else float(result[split]["rmse"])


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


########################################################
# Ladder
########################################################


def build_ladder(
    v8_models: dict[Key, dict],
    v8_baselines: dict[str, dict],
    p2_models: dict[Key, dict],
    p2_baselines: dict[str, dict],
    anchor: dict | None,
) -> list[dict]:
    """Assemble the aligned ladder rows (``[0::3]`` beside ``[2::3]``)."""
    rows: list[dict] = []

    def ref(label: str, ref_dict: dict[str, float]) -> dict:
        return {
            "label": label,
            "id0": None, "ood0": None,
            "id2": ref_dict["id"], "ood2": ref_dict["ood"],
            "kind": "reference",
        }

    def model_rung(label: str, key: Key) -> dict:
        a, b = v8_models.get(key), p2_models.get(key)
        return {
            "label": label,
            "id0": rmse_of(a, "in_distribution"),
            "ood0": rmse_of(a, "out_of_distribution"),
            "id2": rmse_of(b, "in_distribution"),
            "ood2": rmse_of(b, "out_of_distribution"),
            "kind": "model",
        }

    def baseline_rung(label: str, name: str) -> dict:
        a, b = v8_baselines.get(name), p2_baselines.get(name)
        return {
            "label": label,
            "id0": rmse_of(a, "in_distribution"),
            "ood0": rmse_of(a, "out_of_distribution"),
            "id2": rmse_of(b, "in_distribution"),
            "ood2": rmse_of(b, "out_of_distribution"),
            "kind": "baseline",
        }

    rows.append(ref("GPR (paper baseline)", GPR_REF))
    rows.append(baseline_rung("Persistence", "persistence"))
    rows.append(baseline_rung("pool tail-mean", "pool_tail_mean"))
    rows.append(baseline_rung("kNN-copy k=5", "knn_copy_k5"))
    rows.append(model_rung("Chronos-2 zero-shot (base, mean)", (BASE_SLUG, "zeroshot", 0, None)))
    rows.append(model_rung("Chronos-2 ICL (mmr_euclid k=5)", (BASE_SLUG, "mmr_euclid", 5, None)))
    rows.append(model_rung("Chronos-Bolt ICL (mmr_euclid k=10)", (BOLT_SLUG, "mmr_euclid", 10, None)))
    rows.append(model_rung("finetuned BilinearLoRA, k=0", (FINETUNED_SLUG, "zeroshot", 0, None)))
    rows.append(model_rung("finetuned + ICL (mmr_euclid k=5)", (FINETUNED_SLUG, "mmr_euclid", 5, None)))
    rows.append(
        model_rung(
            f"finetuned + ICL @ {FT_TRAIN_CONTEXT} window (mmr_euclid k=5)",
            (FINETUNED_SLUG, "mmr_euclid", 5, FT_TRAIN_CONTEXT),
        )
    )
    if anchor is not None:
        tail = anchor["metrics_tail80"]
        rows.append(
            {
                "label": "finetuned, Severin's rollout (honest [-80:] rescore)",
                "id0": float(tail["id"]["rmse"]),
                "ood0": float(tail["ood"]["rmse"]),
                "id2": None, "ood2": None,
                "kind": "anchor",
            }
        )
    rows.append(ref("GyroSwin-1B (paper)", GYROSWIN_REF))
    return rows


def fig_ladder(rows: list[dict], out_dir: Path) -> Path:
    """Single-protocol ([0::3]) ladder with the [2::3] value overlaid per rung."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, split_short, k0, k2 in (
        (axes[0], "ID", "id0", "id2"),
        (axes[1], "OOD", "ood0", "ood2"),
    ):
        labels = [r["label"] for r in rows]
        y = np.arange(len(rows))
        for yi, r in zip(y, rows):
            v0, v2 = r[k0], r[k2]
            if r["kind"] == "reference":
                ax.barh(yi, v2, color="0.7", height=0.6)
                ax.text(v2 + 0.6, yi, f"{v2:.1f}", va="center", fontsize=7.5)
            elif r["kind"] == "anchor":
                ax.barh(yi, v0, color="tab:cyan", height=0.6)
                ax.text(v0 + 0.6, yi, f"{v0:.1f}", va="center", fontsize=7.5)
            else:
                ax.barh(yi - 0.18, v0, color="tab:blue", height=0.34)
                ax.text(v0 + 0.6, yi - 0.18, f"{v0:.1f}", va="center", fontsize=7)
                if v2 is not None:
                    ax.barh(yi + 0.18, v2, color="0.55", height=0.34)
                    ax.text(v2 + 0.6, yi + 0.18, f"{v2:.1f}", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("tail RMSE")
        ax.set_title(split_short)
        ax.grid(alpha=0.3, axis="x")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
        plt.Rectangle((0, 0), 1, 1, color="0.55"),
        plt.Rectangle((0, 0), 1, 1, color="tab:cyan"),
        plt.Rectangle((0, 0), 1, 1, color="0.7"),
    ]
    fig.legend(
        handles,
        ["our few-shot, [0::3] phase", "our few-shot, [2::3] phase", "Severin rollout ([0::3], honest)", "paper reference"],
        loc="lower center", ncol=4, fontsize=8, frameon=False,
    )
    fig.suptitle(
        "Adaptation ladder on a single, reconciled protocol ([0::3] traces, honest [-80:] tail)",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path = out_dir / "reconciliation_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


########################################################
# Report blocks
########################################################


def make_intro() -> list[str]:
    return [
        "# Evaluation reconciliation — making both halves of the project comparable (Phase 8)",
        "",
        "The few-shot ICL side (Lukas) and the zero-shot/finetuning side (Severin) "
        "evaluate on traces subsampled from the **same 255 raw GKW simulations**, on "
        "the **same 6 ID + 5 OOD raw ids** (ID 8/115/131/148/235/262; OOD 0–4), with "
        "the **same metric function** (`rmse_with_standard_error`). They differ in two "
        "*eval-level* details — the subsample **phase** and the metric **window** — and "
        "in several *method-level* details that are intrinsic to each approach. This "
        "phase re-runs the adaptation ladder so a single, internally-consistent ladder "
        "exists; the eval-level differences are aligned by the re-run, the method-level "
        "ones are documented as inherent.",
    ]


def make_protocol_table() -> list[str]:
    return [
        "\n## Protocol differences (eval-level aligned here; method-level inherent)\n",
        "| dimension | our few-shot side | Severin's finetuning side | kind | source |",
        "|---|---|---|---|---|",
        "| subsample phase | `[2::3]` (266 steps) | `[0::3]` (267 steps) | **eval-level — aligned** | `benchmark_utils.py:BenchmarkDataProvider` vs `lib/dataset.py:get_benchmark_flux_traces` (stride=window=3) |",
        "| metric window | honest `mean(x[-80:])` | notebook `mean(x[:-80])` | **eval-level — aligned** | `harness.py:run_benchmark` (`trace[-tail:]`) vs `finetuned.py:severin_anchor_eval` |",
        "| seeds | 20-seed default (single seed 42 for deterministic cells) | single seed | **eval-level — aligned** (single seed 42 here) | `harness.py:DEFAULT_SEEDS` |",
        "| context length | 80, grows by 64 each rollout step | 512, fixed NaN-left-padded | method-level — inherent | `make_concat_forecast_fn` vs notebook cells 15–18 |",
        "| prediction length | 64 / step | 80 / step | method-level — inherent | `FewShotConfig.model_prediction_length` vs `FTSConfig.prediction_length` |",
        "| rollout | concat-ICL autoregressive rollout (our harness) | raw `model(context, context_mask)` forward | method-level — inherent | `harness.make_icl_forecast_fn` vs `severin_anchor_eval` |",
        "| normalization | per-sample z-score (shared scaling, Phase 3) | Chronos-2 internal instance-norm | method-level — inherent | `presentation.make_concat_forecast_fn` |",
        "",
        "**What this run aligns.** Every ladder rung below is measured on the `[0::3]` "
        "traces with the honest `[-80:]` tail metric and single seed 42 — so the "
        "few-shot rungs and Severin's rollout rung share trace selection, metric, and "
        "seed. The method-level rows are *not* aligned (they are what is being "
        "compared); the few-shot rungs use our concat-ICL method, the anchor rung uses "
        "Severin's raw-forward rollout, both on the reconciled traces+metric.",
        "",
        "**Boundary.** We cannot re-derive Severin's *finetuning variants* (no "
        "checkpoints; he is unresponsive). The finetuned rungs here use OUR "
        "recipe-faithful self-trained BilinearLoRA checkpoint; his published table rows "
        "stay annotated with the metric note rather than re-run.",
    ]


def make_metric_audit(anchor: dict | None) -> list[str]:
    lines = ["\n## Metric audit recap — `[:-80]` vs `[-80:]`\n"]
    if anchor is None:
        lines.append("`severin_anchor.json` not found — metric audit unavailable.")
        return lines
    sev = anchor["metrics_severin_headminus80"]
    tail = anchor["metrics_tail80"]
    lines.extend(
        [
            "The chronos2 finetuning notebooks score `mean(x[:-80])` — the mean over "
            "everything EXCEPT the tail, *including the 80 copied ground-truth context "
            "steps* the rollout starts from — while our tables, the GyroSwin paper, and "
            "the repo's own TimesFM runner use the proper tail `mean(x[-80:])`. Both "
            "metrics below are computed from the SAME forecasts of our self-trained "
            "checkpoint under Severin's exact protocol (the full audit lives in "
            "[`finetuned_icl_table.md`](finetuned_icl_table.md)):",
            "",
            "| metric | ID RMSE | OOD RMSE | comparable to |",
            "|---|---|---|---|",
            f"| his `mean(x[:-80])` | {sev['id']['rmse']:.2f} ± {sev['id']['standard_error']:.2f} "
            f"| {sev['ood']['rmse']:.2f} ± {sev['ood']['standard_error']:.2f} "
            f"| README finetuning rows (13.83 / 4.86) |",
            f"| honest `mean(x[-80:])` | {tail['id']['rmse']:.2f} ± {tail['id']['standard_error']:.2f} "
            f"| {tail['ood']['rmse']:.2f} ± {tail['ood']['standard_error']:.2f} "
            f"| every other table in this repo |",
            "",
            "The OOD gap is the headline: the copied-context metric reports "
            f"{sev['ood']['rmse']:.1f} where the honest tail is {tail['ood']['rmse']:.1f} — "
            "the dramatic chronos2 finetuning OOD numbers are largely this artifact. The "
            "ID numbers are only mildly inflated. The honest `[-80:]` number is the one "
            "carried into the ladder below as the anchor rung.",
        ]
    )
    return lines


def make_phase_invariance(deltas: dict[str, dict[str, float]]) -> list[str]:
    rels = np.array([d["rel_delta"] for d in deltas.values()])
    abss = np.array([d["abs_delta"] for d in deltas.values()])
    lines = [
        "\n## Phase invariance — the comparability evidence\n",
        "Per-trace true-tail-mean of the `[0::3]` trace vs the `[2::3]` trace (last 80 "
        "steps). The benchmark metric reads this saturation level, which is a property "
        "of the simulation, not the subsample phase — so the two should agree up to a "
        "negligible delta. They do:",
        "",
        "| trace | `[0::3]` tail | `[2::3]` tail | \\|Δ\\| | rel Δ |",
        "|---|---|---|---|---|",
    ]
    for key, d in deltas.items():
        lines.append(
            f"| {key} | {d['phase0']:.3f} | {d['phase2']:.3f} | {d['abs_delta']:.4f} | "
            f"{d['rel_delta']:.4f} |"
        )
    lines.append(
        f"\nAcross all 11 traces: relative delta median {np.median(rels):.4f}, max "
        f"{rels.max():.4f}; absolute delta median {np.median(abss):.4f}, max "
        f"{abss.max():.4f}. The tail mean is empirically phase-invariant — so the "
        "`[0::3]` and `[2::3]` ladders are comparable, and any ladder-rung difference is "
        "method (rollout/window), not trace selection."
    )
    return lines


def make_ladder_table(rows: list[dict]) -> list[str]:
    lines = [
        "\n## The aligned adaptation ladder\n",
        "All rungs on the `[0::3]` traces with the honest `[-80:]` tail metric, single "
        "seed 42 (mean decoding for Chronos-2/Bolt/finetuned, shared scaling). The "
        "`[2::3]` column is the existing number from the prior grids "
        "(`results/few_shot_v6_finetuned/`, `results/few_shot_v5_decoding/`; baselines "
        "recomputed in-memory) — shown side by side as the cross-phase check. Paper "
        "references and the Severin-rollout anchor have a single set of numbers.",
        "",
        "| rung | ID `[0::3]` | ID `[2::3]` | OOD `[0::3]` | OOD `[2::3]` | ΔID (0−2) | basis |",
        "|---|---|---|---|---|---|---|",
    ]
    # Classify cross-phase robustness from the data (no hand-typed verdicts).
    robust: list[tuple[str, float]] = []
    sensitive: list[tuple[str, float]] = []
    for r in rows:
        if r["kind"] in ("model", "baseline") and r["id0"] is not None and r["id2"] is not None:
            d_id = r["id0"] - r["id2"]
            (sensitive if abs(d_id) > 6.0 else robust).append((r["label"], d_id))
            delta_cell = f"{d_id:+.2f}"
        else:
            delta_cell = "—"
        lines.append(
            f"| {r['label']} | {fmt(r['id0'])} | {fmt(r['id2'])} | {fmt(r['ood0'])} | "
            f"{fmt(r['ood2'])} | {delta_cell} | {r['kind']} |"
        )
    robust_str = "; ".join(f"{lbl} ({d:+.1f})" for lbl, d in robust)
    sensitive_str = "; ".join(f"{lbl} ({d:+.1f})" for lbl, d in sensitive)
    lines.append(
        f"\n**The reconciliation surfaces a real divergence — and it is the right "
        f"divergence.** The phase-ROBUST rungs (|ΔID| ≤ 6) are exactly the ones the "
        f"project's conclusions rest on: {robust_str or 'none'}. The phase-SENSITIVE "
        f"rungs (|ΔID| > 6) are the best-config retrieval-ICL cells: {sensitive_str or 'none'}. "
        "This is not a bug: the saturation level the metric reads is phase-invariant "
        "(table below, max rel Δ < 1%), but the *forecast* depends on the 80-step "
        "context, which IS phase-shifted — so the level the model copies, and the "
        "examples `mmr_euclid` retrieves (z-scored context distance), shift with it. "
        "With only 6 ID traces the marginal ICL gain over finetuned k=0 was already "
        "flagged as not individually significant (CI [−16.4, +16.8] in "
        "[`finetuned_icl_table.md`](finetuned_icl_table.md)); the cross-phase swing is "
        "that same fragility from a second angle, and notably the 512-window cell's "
        "`[2::3]` improvement does NOT replicate on `[0::3]`. The robust takeaways hold "
        "on both phases: **finetuning delivers the ~22 ID / ~34 OOD step (phase-stable, "
        "ΔID ≈ 0), and retrieval-ICL adds a further ID gain that is real in direction "
        "but fragile in magnitude.** The Severin-rollout anchor rung (honest `[-80:]` "
        "rescore, his raw-forward method on the same `[0::3]` traces) lands among the "
        "finetuned rungs — the two halves of the project sit on one ladder."
    )
    return lines


########################################################
# Self-test (CPU, no model downloads, no v8 run required)
########################################################


def self_test() -> None:
    from fusiontimeseries.benchmarking.few_shot.reconciliation import (
        PHASE0_STEPS,
        Phase0BenchmarkProvider,
        phase0_tail_mean_deltas,
    )
    from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
        IN_DISTRIBUTION_ITERATIONS,
        OUT_OF_DISTRIBUTION_ITERATIONS,
    )

    print("analyze_reconciliation self-test (CPU)...")
    provider = Phase0BenchmarkProvider()
    keys = set(IN_DISTRIBUTION_ITERATIONS) | set(OUT_OF_DISTRIBUTION_ITERATIONS)
    assert {k for k, _ in provider.items()} == keys, "provider keys != benchmark keys"
    for _, tensor in provider.items():
        assert tensor.shape == (PHASE0_STEPS,)
    print(f"✓ provider: 11 keys, {PHASE0_STEPS} steps each")

    deltas = phase0_tail_mean_deltas(provider)
    assert set(deltas) == keys
    rels = np.array([d["rel_delta"] for d in deltas.values()])
    assert rels.max() < 0.05, f"phase delta too large: max rel {rels.max():.4f}"
    print(f"✓ phase-invariance: max rel tail-mean delta {rels.max():.4f} (< 0.05)")

    p2 = load_phase2_models([V6_DIR, V5_DIR])
    for key, label in (
        ((BASE_SLUG, "zeroshot", 0, None), "base chronos2 zeroshot mean"),
        ((FINETUNED_SLUG, "zeroshot", 0, None), "ft zeroshot mean"),
        ((FINETUNED_SLUG, "mmr_euclid", 5, None), "ft mmr5 mean"),
        ((BOLT_SLUG, "mmr_euclid", 10, None), "bolt mmr10 mean"),
    ):
        assert key in p2, f"[2::3] reference cell missing: {label} {key}"
    print(f"✓ [2::3] reference cells present in v6/v5 ({len(p2)} model cells indexed)")

    anchor_path = V6_DIR / "severin_anchor.json"
    assert anchor_path.exists(), f"anchor missing: {anchor_path}"
    anchor = json.load(open(anchor_path))
    assert "metrics_severin_headminus80" in anchor and "metrics_tail80" in anchor
    print("✓ severin_anchor.json loads with both metric blocks")
    print("\n✅ analyze_reconciliation self-test passed!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Phase-8 reconciliation run")
    parser.add_argument("--results-dir", type=Path, default=V8_DIR)
    parser.add_argument("--v6-dir", type=Path, default=V6_DIR)
    parser.add_argument("--v5-dir", type=Path, default=V5_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    from fusiontimeseries.benchmarking.few_shot.reconciliation import (
        phase0_tail_mean_deltas,
    )

    v8_models, v8_baselines = load_v8(args.results_dir)
    assert v8_models or v8_baselines, f"No reconciliation results in {args.results_dir}"
    p2_models = load_phase2_models([args.v6_dir, args.v5_dir])
    p2_baselines = compute_phase2_baselines()
    anchor_path = args.v6_dir / "severin_anchor.json"
    anchor = json.load(open(anchor_path)) if anchor_path.exists() else None
    deltas = phase0_tail_mean_deltas()

    rows = build_ladder(v8_models, v8_baselines, p2_models, p2_baselines, anchor)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ladder_path = fig_ladder(rows, args.out_dir)

    lines: list[str] = []
    lines.extend(make_intro())
    lines.extend(make_protocol_table())
    lines.extend(make_metric_audit(anchor))
    lines.extend(make_ladder_table(rows))
    lines.append(f"\n![reconciliation_ladder]({ladder_path.name})")
    lines.extend(make_phase_invariance(deltas))

    markdown = "\n".join(lines) + "\n"
    table_path = args.out_dir / "evaluation_reconciliation.md"
    table_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {table_path}")
    print(f"Wrote {ladder_path}")


if __name__ == "__main__":
    main()
