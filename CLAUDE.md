# Claude Assistant Guide for FusionTimeSeries

**Project**: Heat Flux Time-Series Prediction for Gyrokinetic Plasma Turbulence
**Version**: 1.0.0
**Python**: >=3.13
**Package Manager**: uv

---

## Project Overview

Master's thesis project predicting heat flux values with pre-trained time-series
foundation models (TiRex, Chronos-2, TimesFM) on pre-generated GKW gyrokinetic
plasma turbulence simulation data. Covers zero-shot benchmarking, few-shot
in-context learning, and finetuning (LoRA and bilinear LoRA variants), with
in-distribution (ID) vs out-of-distribution (OOD) evaluation.

**Fork note**: This repo (`lukaskurz/fusiontimeseries`) is a fork of
`sbergsmann/fusiontimeseries` (remote `upstream`). Upstream's v1.0.0
restructuring was merged in June 2026. Our own contributions on top of
upstream: `benchmarking/zero_shot/`, `benchmarking/few_shot/`, the `mps`
extra, and the kept `finetuning/preprocessing/utils.py` and
`data/flux/benchmark/flux_data.json` (upstream deleted both, but our
benchmarking code depends on them).

**What this is NOT:**
- Not reimplementing GyroSwin architecture
- Not running gyrokinetic simulations (data is pre-generated)
- Not implementing 5D spatial models (1D time-series only)

---

## Quick Start

```bash
# Install dependencies (pick one)
uv sync --extra cpu --group dev    # CPU
uv sync --extra cu126 --group dev  # CUDA 12.6
uv sync --extra mps --group dev    # Apple Silicon (our addition)
```

Environment variables (`.env`, see `.env.example`):
```bash
FLUX_TRACE_DIR=/path/to/data/flux/raw
BENCHMARK_SAVE_DIR=/path/to/benchmark/results
```

Pre-commit hooks are currently disabled upstream (`.pre-commit-config.yaml`
has `repos: []`); ruff/nbstripout entries are commented out.

There are no CLI entry points anymore (the old `b-*-v2` scripts were removed
along with the legacy namespace). Work happens in notebooks, plus
`uv run src/fusiontimeseries/experiments/main.py` for the finetuning
experiment runner.

---

## Architecture Overview

```
src/fusiontimeseries/
├── benchmarking/              # OUR work (not in upstream)
│   ├── zero_shot/             # Zero-shot benchmarks
│   │   ├── benchmark_utils.py # BenchmarkDataProvider, BenchmarkConfig,
│   │   │                      #   rmse_with_standard_error, Utils
│   │   └── *_benchmark.ipynb
│   └── few_shot/              # Few-shot ICL benchmarks (k example traces)
│       ├── few_shot_utils.py  # FewShotExample, create_example_pool
│       │                      #   (excludes by RAW id; pool = 245 traces)
│       ├── operating_params.py            # raw/benchmark <-> gyroswin-dump
│       │                                  #   mapping + q/shat/rlt/rln lookup
│       ├── operating_params_mapping.json  # tracked; rebuild via module __main__
│       ├── harness.py         # run_benchmark (multi-seed), paired_comparison,
│       │                      #   make_icl_forecast_fn, load_results/results_table
│       ├── baselines.py       # persistence / pool tail-mean / kNN-copy
│       ├── selection.py       # Phase-2 retrieval strategies: STRATEGIES,
│       │                      #   make_select_fn (op_knn / ctx_euclid / ctx_dtw /
│       │                      #   ctx_growth / oracle_tail / mmr_euclid)
│       ├── rerun_ksweep.py    # fixed-pool re-run of the old k-sweeps; also
│       │                      #   MODEL_SLUGS + PREDICT_FACTORIES model wrappers
│       │                      #   + make_chronos2_pipeline factory
│       ├── run_selection_grid.py  # Phase-2 grid: strategy x k x model
│       ├── analyze_selection.py   # grid analysis -> docs/results/fewshot/
│       ├── presentation.py    # Phase-3 presentation variants: concat fn with
│       │                      #   per_example/shared norm, chronos2 group ICL,
│       │                      #   ordering + truncation SelectFn wrappers
│       ├── run_presentation_grid.py  # Phase-3 staged grid (A group / B norm /
│       │                             #   C order / D trunc); skip-if-exists
│       ├── analyze_presentation.py   # -> docs/results/fewshot/presentation_*
│       ├── covariates.py      # Phase-4 training-free OP conditioning
│       │                      #   (chronos2 covariates): step channels over the
│       │                      #   concat stream, by-value query resolver,
│       │                      #   permuted control, group+cov; affine-erasure
│       │                      #   theory in the module docstring
│       ├── run_covariates_grid.py    # Phase-4 grid (anchors/main/k1/perm/
│       │                             #   group) + S1-S5 smoke (tri-state checks)
│       ├── analyze_covariates.py     # -> docs/results/fewshot/covariates_*
│       │                             #   + adaptation_ladder.png
│       ├── run_decoding_grid.py      # Phase-5 grid: point_stat median/mean
│       │                             #   (/meanhead TimesFM) x anchors/best/
│       │                             #   oracle/random + D1-D4 smoke
│       ├── analyze_decoding.py       # -> docs/results/fewshot/decoding_table.md
│       │                             #   + decoding_effect.png; seed/model
│       │                             #   ensembling (post-hoc over tail means)
│       ├── finetuned.py       # Phase-6 finetuned-model wrappers: checkpoint
│       │                      #   reconstruction (load_finetuned_chronos2,
│       │                      #   window knob 8192/512), raw_param_tensor
│       │                      #   (FluxData order [shat,q,rlt,rln]!),
│       │                      #   make_finetuned_forecast_fn (per-query
│       │                      #   ConditionRegistry patch), severin_anchor_eval
│       │                      #   (his [:-80] metric + honest [-80:])
│       ├── run_finetuned_grid.py     # Phase-6 grid: {base, ft} x configs x
│       │                             #   {median, mean} + ft win512 + anchor
│       │                             #   stage + F1-F6 smoke (F2b param-order
│       │                             #   go/no-go gate)
│       ├── analyze_finetuned.py      # -> docs/results/fewshot/
│       │                             #   finetuned_icl_table.md +
│       │                             #   finetuned_synergy.png + regenerated
│       │                             #   adaptation_ladder.png
│       └── *_fewshot_benchmark.ipynb
├── zeroshot/                  # Upstream's zero-shot notebooks (v1.0.0)
├── finetuning/
│   ├── chronos2/              # Chronos-2 LoRA/full/bilinear notebooks + dataset.py
│   │                          #   + OUR train_bilinear.py (Phase-6: script-ified
│   │                          #   BilinearLoRA recipe, MPS adaptations,
│   │                          #   ensure_flat_flux_data — rebuilds the flat-list
│   │                          #   flux_data schema lib/dataset.py expects)
│   ├── timesfm/               # TimesFM finetuning notebooks + dataset.py, trainer.py
│   ├── evaluation/            # loss_curve.py
│   └── preprocessing/         # OUR kept module: utils.py (get_valid_flux_traces)
├── experiments/               # Upstream finetuning experiment runner
│   ├── main.py                # entry point (uv run .../experiments/main.py)
│   ├── config.py, dataset.py, model.py, trainer.py, patch.py
│   └── analysis/plotting scripts
├── ablations/                 # Normalization / LR ablation studies
├── lib/                       # Shared utilities
│   ├── benchmarking.py        # rmse_with_standard_error (upstream copy)
│   ├── config.py              # FTSConfig; reads data/flux_data.json
│   ├── conditioning.py        # ConditionRegistry (operating-param conditioning)
│   └── dataset.py, modules.py, get_next_path.py
└── loralib/                   # Adapted Microsoft loralib
    └── layers.py              # LoRA, BilinearLoRA, OSS/RSSBilinearLoRA

playground/                    # Quick demo notebooks (utils.py has plot_forecast)
docs/
├── methods/                   # BilinearLoRA / OSS / RSS method docs
├── results/                   # zeroshot + fewshot + finetuning result plots
│   └── fewshot/               # Phase-2 selection table + figures
├── report/, poster/
└── installation.md
data/
├── flux_data.json             # Upstream data dump: batch_1..batch_10 +
│                              #   gyroswin_{train,val,id,ood} splits
├── stable_simulations.json
├── download_batches.py, download_gyroswin_data.py, prepare_simulation_data.py
└── flux/                      # gitignored (data/flux in .gitignore)
    ├── raw/fluxes_{0..299}.dat   # untracked, kept on disk locally
    └── benchmark/flux_data.json  # OUR benchmark data (tracked, force-kept):
                                  #   in_distribution (6) / out_of_distribution (5)
```

---

## Key Components (our benchmarking pipeline)

```python
from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
    BenchmarkDataProvider,
    BenchmarkConfig,
    rmse_with_standard_error,
    Utils,
)

provider = BenchmarkDataProvider()           # reads data/flux/benchmark/flux_data.json
id_trace = provider.get_id("iteration_8_ifft")          # torch.Tensor [266]
ood_trace = provider.get_ood("ood_iteration_0_ifft_realpotens")
rmse, se = rmse_with_standard_error(y_true, y_pred)
```

Few-shot utilities (`benchmarking/few_shot/few_shot_utils.py`) build k-shot
contexts from training traces; they import `get_valid_flux_traces` from
`fusiontimeseries.finetuning.preprocessing.utils` (requires `FLUX_TRACE_DIR`
pointing at the raw `.dat` files).

Upstream's pipeline (`lib/`, `experiments/`, `ablations/`) instead reads the
big `data/flux_data.json` dump via `lib/config.py:FLUX_DATA_PATH`.

---

## Data Notes

- Raw traces: `fluxes_{iteration}.dat` (301 files, 0..300), 3-column ASCII
  (electron, energy, ion flux); only column 1 (energy flux) is used.
  800 timesteps; pool traces are `[::3]` (267 steps), benchmark traces are
  `[2::3]` (266 steps) of the same simulations.
- 251 of 301 raw traces are valid for training (mean flux ≥ 1.0 at head/tail).
- Operating params (q, shat, rlt, rln) live only in `data/flux_data.json`
  under PERMUTED gyroswin keys (1000+i ≠ raw i). The verified value-matched
  mapping is `few_shot/operating_params_mapping.json` (tracked); use the
  lookup API in `few_shot/operating_params.py`, never assume key order.
- `create_example_pool(exclude_ids=...)` takes RAW iteration ids (since
  2026-06-11; before that it dropped pool positions and leaked the six ID
  test twins into the pool — see the README few-shot note).
- Context length 80 (linear phase), prediction length 64, evaluation on last
  80 timesteps.
- `data/flux` is gitignored. The raw `.dat` files and our benchmark JSON are
  kept on disk locally; the benchmark JSON is force-tracked
  (`git add -f data/flux/benchmark/flux_data.json` if it ever needs re-adding).
- Upstream's `data/flux_data.json` (tracked, large) has a DIFFERENT schema
  than our benchmark JSON — batches and gyroswin splits keyed by iteration id,
  not in/out-of-distribution keys. Don't confuse the two.
- THIRD schema gotcha: `lib/dataset.py:load_flux_data` (used by the
  finetuning notebooks' `Chronos2Dataset`) expects an older FLAT-LIST
  flux_data schema (FluxData records with idx/distribution) that no longer
  exists in the repo. `finetuning/chronos2/train_bilinear.py:
  ensure_flat_flux_data()` rebuilds it from the gyroswin dump (raw ids via
  the operating-params mapping) into `data/flux/flux_data_flat.json`
  (gitignored); set `FTSConfig.data_path` to it.
- CONDITIONING ORDER: FluxData / the finetuned BilinearLoRA conditioning use
  the `lib/config.py:OP_NAMES` order **[shat, q, rlt, rln]**; the few-shot
  `operating_params.OP_NAMES` is (q, shat, rlt, rln). Use
  `finetuned.raw_param_tensor` (built from the lib order); smoke F2b gates it.

---

## Important Conventions

- **Tensor shapes**: traces `[timesteps]`; context `[batch, 1, context_length]`;
  forecasts `[batch, prediction_length, n_quantiles]`. Some models return
  `[batch, n_quantiles, pred_len]` — normalize orientation in your pipeline.
- **Normalization**: always per-sample z-score; keep predictions normalized
  through the pipeline and denormalize only for evaluation.
- **Device handling differs per model**: Chronos wants CPU input (handles
  device internally), TiRex wants device-placed input. The `mps` extra exists
  for Apple Silicon; `playground/utils.py` uses `.cpu().numpy()` before
  plotting for this reason.
- **Style**: type hints required, Google-style docstrings, ruff formatting,
  absolute imports from `fusiontimeseries.*`. Seeds set to 42.

---

## Known Issues & Gotchas

1. `FLUX_TRACE_DIR` / `BENCHMARK_SAVE_DIR` must be set or scripts crash.
2. Upstream's `zeroshot/` notebooks were committed referencing the deleted
   `fusiontimeseries.benchmarking.benchmark_utils` path; we repointed all
   notebooks to `fusiontimeseries.benchmarking.zero_shot.benchmark_utils`.
3. Pre-commit hooks are disabled — notebook outputs are no longer stripped
   automatically; be careful not to commit huge outputs.
4. `tirex-ts` is installed with the `cuda` extra per upstream's pyproject;
   on macOS the CUDA-specific bits are inert. Running TiRex on MPS/CPU needs
   `TIREX_NO_CUDA=1` set BEFORE the first forward pass (else the sLSTM
   CUDA-kernel JIT crashes with `KeyError: 'CUDA_LIB'`);
   `rerun_ksweep.make_tirex_predict` sets it automatically.
5. `timesfm` must resolve to >= 2.0.1 for the `TimesFM_2p5_200M_torch` /
   `ForecastConfig` API our notebooks use; the pyproject pin `>=1.3.0` also
   allows 1.3.0 (old 1.x API) — check `uv.lock` after upstream merges.
6. No automated tests exist. Smoke tests after changes:
   `uv run --no-sync python -c "from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import BenchmarkDataProvider; print(BenchmarkDataProvider().get_id('iteration_8_ifft').shape)"`
   (expect `torch.Size([266])`), plus the module self-tests
   (`python -m fusiontimeseries.benchmarking.few_shot.{operating_params,few_shot_utils,harness,baselines,selection}`).

---

## Current Results

See `README.md` for the full, current tables: upstream's zero-shot results
(TiRex best: ID 79.49 ± 14.38), our few-shot ICL results (fixed-pool re-run
2026-06-11; best: Chronos-Bolt-Tiny k=10 at ID 30.65 / OOD 34.62, dirs
`results/few_shot_v2*`), model-free baselines (kNN-copy k=5: ID 34.98), and
upstream's finetuning results. Phase-2 example-selection grid
(`results/few_shot_v2_selection/`, analysis in
`docs/results/fewshot/selection_table.md`): retrieval does NOT beat random
on ID (even the cheating oracle doesn't, for 3/4 models — per-example
z-scoring hides the level signal → Phase 3); small consistent OOD gains;
op_knn does not beat context-NN. Phase-3 presentation grid
(`results/few_shot_v3_presentation/`, analysis in
`docs/results/fewshot/presentation_table.md`): SHARED scaling (one query-fit
scaler for examples+query) is the fix — oracle_tail finally works (TimesFM
15.99 ID / 8.10 OOD at k=10, significant vs random__shared on all 4 models),
random__shared gets WORSE (wrong levels transfer), best legit ID improves
30.65 → 23.28 (Bolt mmr_euclid shared k=10); Chronos-2 group ICL is much
worse than concat; ordering is a non-factor; truncation backfires under
shared norm. Phase-4 covariate grid (`results/few_shot_v4_covariates/`,
analysis in `docs/results/fewshot/covariates_table.md` +
`adaptation_ladder.png`): training-free OP conditioning via Chronos-2
covariate channels is a clean NEGATIVE — Chronos-2 instance-norms each row
independently, so constant channels are value-erased exactly (verified
bit-identical for tri-state-matched values; the surviving float32 tri-state
artifact still perturbs outputs, and is device-dependent CPU vs MPS);
step-encoded channels are ≈ the permuted-params control everywhere
(presence, not information), homogenizing all strategies toward ~47 ID
(zeroshot+cov anchor improves 109.91 → 81.02 ID carrying provably no
information; oracle k=10 destroyed 23.31 → 47.28). The adaptation ladder
(zero-shot 109.91 → ICL 29.40 → +cov 44.61 → finetuned 13.83 ID) is the
bridge deliverable. Phase-5 decoding grid (`results/few_shot_v5_decoding/`,
analysis in `docs/results/fewshot/decoding_table.md` +
`decoding_effect.png`): mean decoding (decile average over the 9 quantiles;
NOTE TiRex has NO native mean — the library's "mean" return is a relabeled
median) improves ID in 14/16 cells, most where calibration is worst
(Chronos-2 zero-shot −20.4, random −5.2 sig.); new best legit ID 22.63
(Bolt mmr_euclid shared k=10 + mean). Per-model decoding default going
forward: mean for Chronos-2/Bolt/TiRex, MEDIAN for TimesFM (its decile
mean +1.05 and native meanhead +2.01 ID are significantly worse at the
best config). Seed-ensembling (average the 20 random-set forecasts before
scoring) is significantly better everywhere but loses to retrieval;
cross-model ensembling never beats the best single model. Phase-6
finetuned-ICL grid (`results/few_shot_v6_finetuned/`, analysis in
`docs/results/fewshot/finetuned_icl_table.md` + `finetuned_synergy.png` +
regenerated `adaptation_ladder.png`; SELF-TRAINED BilinearLoRA checkpoint
via `train_bilinear.py`, Severin's exact recipe, sha256 in every JSON —
his weights swap in via `--checkpoint`): adaptation STACKS on ID through
retrieval quality — ft k=0 22.20 ID (mean) already beats best base ICL
(27.06), ft+mmr k=5 18.62, clamped to the 512 TRAINING window **15.63 ID =
project best legit** (marginal gain n.s. at n=6; direction consistent);
oracle stacks significantly (9.39 ID/10.89 OOD) → ICL capacity survives
finetuning, retrieval is the bottleneck; random examples erase the ft
advantage; OOD is finetuning-only (67.94 → 34.10). METRIC AUDIT: the
chronos2 finetuning notebooks score mean(x[:-80]) INCLUDING the 80 copied
context steps — README 13.83/4.86 is on that easier metric; honest [-80:]
rescore of the same forecasts: ID 17.51 / OOD 40.64 (OOD advantage largely
artifact; note-for-Severin in the table doc). Robustness (2026-06-13):
step-4000 weights worse everywhere than the shipped step-200 best-eval
pick (results/few_shot_v6_finetuned_step4000/ — overtraining degrades ICL
most); oracle@win512 19.57 vs 9.39 full → the win512 gain is context
COMPOSITION (clamp drops wrong-level example mass), not window-length
mismatch.
`docs/results/` has plots; `docs/methods/`
documents the Bilinear/OSS/RSS LoRA variants.

---

## Syncing with upstream

```bash
git fetch upstream
git merge upstream/main
```
When merging, protect our kept files if upstream touches/deletes them again:
`benchmarking/`, `finetuning/preprocessing/`, `data/flux/benchmark/flux_data.json`,
the `mps` extra in `pyproject.toml`. Regenerate `uv.lock` with `uv lock`
after resolving `pyproject.toml`.

---

**Last Updated**: 2026-06-12 (Phase 0/1: operating-params mapping, pool
leakage fix, evaluation harness, baselines, fixed-pool k-sweep re-runs;
Phase 2: retrieval-based example selection — selection.py, grid + analysis;
Phase 3: example presentation — presentation.py, staged grid + analysis,
shared-scaling headline result; Phase 4: training-free OP conditioning —
covariates.py, staged grid + analysis, negative result + adaptation ladder;
Phase 5: point-stat decoding + ensembling — run_decoding_grid.py + analysis,
per-model mean-vs-median decision, seed/model ensembling verdicts;
Phase 6: ICL x finetuning — train_bilinear.py self-trained BilinearLoRA,
finetuned.py + run_finetuned_grid.py + analysis, synergy verdict +
[:-80] metric audit + corrected adaptation ladder)
**Upstream Maintainer**: Severin Bergsmann (sbergsmann)
