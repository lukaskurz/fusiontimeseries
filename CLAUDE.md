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
│       ├── few_shot_utils.py
│       └── *_fewshot_benchmark.ipynb
├── zeroshot/                  # Upstream's zero-shot notebooks (v1.0.0)
├── finetuning/
│   ├── chronos2/              # Chronos-2 LoRA/full/bilinear notebooks + dataset.py
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
├── results/                   # zeroshot + finetuning result plots
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

- Raw traces: `fluxes_{iteration}.dat`, 3-column ASCII (electron, energy, ion
  flux); only column 1 (energy flux) is used. 800 timesteps, subsampled every
  3rd → 266.
- 251 of 300 raw traces are valid for training (mean flux ≥ 1.0 at head/tail).
- Context length 80 (linear phase), prediction length 64, evaluation on last
  80 timesteps.
- `data/flux` is gitignored. The raw `.dat` files and our benchmark JSON are
  kept on disk locally; the benchmark JSON is force-tracked
  (`git add -f data/flux/benchmark/flux_data.json` if it ever needs re-adding).
- Upstream's `data/flux_data.json` (tracked, large) has a DIFFERENT schema
  than our benchmark JSON — batches and gyroswin splits keyed by iteration id,
  not in/out-of-distribution keys. Don't confuse the two.

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
   on macOS the CUDA-specific bits are inert.
5. No automated tests exist. Smoke test after changes:
   `uv run --no-sync python -c "from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import BenchmarkDataProvider; print(BenchmarkDataProvider().get_id('iteration_8_ifft').shape)"`
   (expect `torch.Size([266])`).

---

## Current Results

See `README.md` for the full, current tables: upstream's zero-shot results
(TiRex best: ID 79.49 ± 14.38), our few-shot ICL results (k=5 TiRex best:
ID 42.33), and upstream's finetuning results. `docs/results/` has plots;
`docs/methods/` documents the Bilinear/OSS/RSS LoRA variants.

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

**Last Updated**: 2026-06-11 (post upstream v1.0.0 merge)
**Upstream Maintainer**: Severin Bergsmann (sbergsmann)
