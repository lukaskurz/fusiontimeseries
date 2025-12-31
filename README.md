# 🔥 Heat Flux Time Series Prediction in Tokamak Reactors

![Python 3.13](https://img.shields.io/badge/python-3.13-blue?style=flat-square&logo=python&logoColor=white)
![UV](https://img.shields.io/pypi/v/uv?label=uv&style=flat-square&logo=pypi&logoColor=white)


Welcome to the Fusion Time Series playground — example code and notebooks for experimenting with TiRex time-series forecasting models and surrounding tooling.


## 🧰 Installation & environment (Windows-focused)

These instructions were tested on Windows (PowerShell). Adjust paths/commands as needed for macOS/Linux.

0. Install [uv](https://docs.astral.sh/uv/#installation)

1. Clone the repo:

```powershell
git clone https://github.com/sbergsmann/fusiontimeseries.git
cd fusiontimeseries
```

2. Install dependencies:

For GPU support (CUDA 12.6):
```powershell
uv sync --extra cu126 --group dev
```

For CPU-only:
```powershell
uv sync --extra cpu --group dev
```

3. Install pre-commit hooks

```powershell
pre-commit install
```

Notes:
- `nbstripout` will clear outputs from `.ipynb` files before they are committed.
- Ruff is configured to attempt automatic fixes (`--fix`) where appropriate.

## 🚀 Quick Test

Run the [TiRex Playbook](playground/tirex-playbook.ipynb) from top to bottom to confirm a working environment.

## 🪟 Only tested on Windows

This repository and the included notebook have been validated on Windows (PowerShell). If you run into platform-specific issues on macOS/Linux, please file an issue or open a PR with repro steps.

## 📊 Results

### Benchmarking Results (Zero-Shot Models)

Comparison of pre-trained time series models on heat flux prediction without finetuning. Metrics measured on the last 80 timesteps (prediction tail).

| Model | In-Distribution RMSE | In-Distribution SE | Out-of-Distribution RMSE | Out-of-Distribution SE | Date |
|-------|---------------------|-------------------|-------------------------|----------------------|------|
| **NX-AI/TiRex** | **63.91** | **13.62** | **44.79** | **7.92** | 2025-12-26 |
| google/timesfm-2.5-200m-pytorch | 82.79 | 11.69 | 62.78 | 14.51 | 2025-12-26 |
| amazon/chronos-2 | 84.86 | 14.18 | 60.78 | 12.75 | 2025-12-26 |
| amazon/chronos-bolt-tiny | 87.78 | 13.76 | 68.02 | 13.00 | 2025-12-26 |

**Configuration**: All models used prediction length of 64 timesteps, starting context length of 80 timesteps, evaluated on 6 in-distribution and 5 out-of-distribution samples.

### Finetuning Results

Performance of finetuned models on the same heat flux prediction task.

| Model | Training Samples | Fine-tune Mode | In-Distribution RMSE | In-Distribution SE | Out-of-Distribution RMSE | Out-of-Distribution SE | Date |
|-------|-----------------|----------------|---------------------|-------------------|-------------------------|----------------------|------|
| **Chronos2 (Finetuned)** | 251 | LoRA | **18.29** | **3.54** | **37.45** | **6.77** | 2025-12-31 |

**Finetuning Configuration**:
- Fine-tune mode: LoRA (Low-Rank Adaptation)
- Learning rate: 5e-5
- Fine-tune steps: 3000
- Batch size: 64
- Prediction length: 64 timesteps
- Context length: 80 timesteps

**Key Findings**:
- 🎯 Finetuning Chronos2 with LoRA achieved **71.4% improvement** in ID RMSE compared to zero-shot (84.86 → 18.29)
- 🎯 **38.4% improvement** in OOD RMSE compared to zero-shot (60.78 → 37.45)
- 📈 TiRex shows best zero-shot performance, but finetuned Chronos2 outperforms all zero-shot models significantly

# Finetuning

## Chronos Finetuning

- https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html?utm_source=chatgpt.com
- Supports Covariates

## TimesFM Finetuning

- https://github.com/pfnet-research/timesfm_fin
-
