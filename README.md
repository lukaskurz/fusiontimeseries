# ⚛️ Flux Time Series Prediction in Tokamak Reactors

![Python 3.13](https://img.shields.io/badge/python-3.13-blue?style=flat-square&logo=python&logoColor=white)
![UV](https://img.shields.io/pypi/v/uv?label=uv&style=flat-square&logo=pypi&logoColor=white)


Welcome to the Fusion Time Series playground — example code and notebooks for experimenting with flux time-series forecasting models and surrounding tooling.


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

| Model                           | In-Distribution RMSE | In-Distribution SE | Out-of-Distribution RMSE | Out-of-Distribution SE | Date       |
| ------------------------------- | -------------------- | ------------------ | ------------------------ | ---------------------- | ---------- |
| **NX-AI/TiRex**                 | **63.91**            | **13.62**          | **44.79**                | **7.92**               | 2025-12-26 |
| google/timesfm-2.5-200m-pytorch | 82.79                | 11.69              | 62.78                    | 14.51                  | 2025-12-26 |
| amazon/chronos-2                | 84.86                | 14.18              | 60.78                    | 12.75                  | 2025-12-26 |
| amazon/chronos-bolt-tiny        | 87.78                | 13.76              | 68.02                    | 13.00                  | 2025-12-26 |

**Configuration**: All models used prediction length of 64 timesteps, starting context length of 80 timesteps, evaluated on 6 in-distribution and 5 out-of-distribution samples.

### Finetuning Results

Performance of finetuned models on the same heat flux prediction task.

300 flux trace simulations were at our disposal. For the following experiments 49 traces were filtered out due to time series head and tail means not being between 1 and inf. The remaining 251 samples were sub-sampled from 800 timesteps to 266 (every 3rd timestep) to resemble the benchmark time series as closely as possible. For finetuning the validation set consists of the tail of

To increase the amount of training data, I subsampled each flux trace three times with a distance of three timesteps.

| Model                               | Training Samples | Fine-tune Mode | In-Distribution RMSE | In-Distribution SE | Out-of-Distribution RMSE | Out-of-Distribution SE | Date       |
| ----------------------------------- | ---------------- | -------------- | -------------------- | ------------------ | ------------------------ | ---------------------- | ---------- |
| Chronos2 (Finetuned)                | 251              | LoRA           | 18.29                | 3.54               | 37.45                    | 6.77                   | 2025-12-31 |
| **Chronos2 (Hyperparameter Tuned)** | 251              | LoRA           | **15.41**            | **2.55**           | **34.05**                | **9.87**               | 2026-01-01 |
| Chronos2 T2 (Augmented Data)        | 753              | LoRA           | 26.32                | 4.53               | 36.54                    | 10.56                  | 2026-01-01 |

**Finetuning Configuration (Hyperparameter Tuned)**:
- Fine-tune mode: LoRA (Low-Rank Adaptation)
- Learning rate: 1.77e-4 (Bayesian optimized)
- Fine-tune steps: 3000
- Batch size: 64 (Bayesian optimized)
- Cross learning: True (Bayesian optimized)
- LoRA config: r=16, alpha=32, dropout=0.1
- Prediction length: 64 timesteps
- Context length: 80 timesteps
- Hyperparameter tuning: 8 trials with Bayesian search

**Key Findings**:
- 🎯 Hyperparameter tuning Chronos2 with LoRA achieved **81.8% improvement** in ID RMSE compared to zero-shot (84.86 → 15.41)
- 🎯 **44.0% improvement** in OOD RMSE compared to zero-shot (60.78 → 34.05)
- 🔧 Hyperparameter tuning improved ID RMSE by **15.8%** over basic finetuning (18.29 → 15.41)
- 🔧 Hyperparameter tuning improved OOD RMSE by **9.1%** over basic finetuning (37.45 → 34.05)
- 📈 TiRex shows best zero-shot performance, but hyperparameter-tuned Chronos2 outperforms all zero-shot models significantly

# Finetuning

## Chronos Finetuning

- https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html?utm_source=chatgpt.com
- Supports Covariates

## TimesFM Finetuning

- https://github.com/pfnet-research/timesfm_fin
-
