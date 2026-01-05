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

### Few-Shot In-Context Learning Results

Performance of pre-trained models using few-shot in-context learning (ICL) without any finetuning. Models are provided with k example traces at inference time to adapt to the task.

**Optimal Configuration (k=5 examples):**

| Model                           | k   | In-Distribution RMSE | In-Distribution SE | Out-of-Distribution RMSE | Out-of-Distribution SE | Improvement vs Zero-Shot | Date       |
| ------------------------------- | --- | -------------------- | ------------------ | ------------------------ | ---------------------- | ------------------------ | ---------- |
| **NX-AI/TiRex**                 | 5   | **42.33**            | **8.95**           | **33.89**                | **14.03**              | **46.4% ID / 45.8% OOD** | 2026-01-05 |
| google/timesfm-2.5-200m-pytorch | 5   | 45.04                | 8.87               | 39.71                    | 17.04                  | 53.8% ID / 54.7% OOD     | 2026-01-05 |
| amazon/chronos-2                | 5   | 46.77                | 8.98               | 41.05                    | 17.06                  | 57.4% ID / 52.2% OOD     | 2026-01-05 |
| amazon/chronos-bolt-tiny        | 1   | 69.15                | 10.77              | 57.05                    | 16.68                  | 38.1% ID / 37.1% OOD     | 2026-01-05 |

**Few-Shot Learning Curve (all models tested with k=0, 1, 3, 5, 10):**

<details>
<summary>Click to expand full k-shot results</summary>

**Chronos-2:**
| k   | ID RMSE | OOD RMSE | ID Δ%  | OOD Δ%  |
| --- | ------- | -------- | ------ | ------- |
| 0   | 109.91  | 85.86    | —      | —       |
| 1   | 75.69   | 61.60    | -31.1% | -28.3%  |
| 3   | 53.60   | 46.71    | -51.2% | -45.6%  |
| 5   | 46.77   | 41.05    | -57.4% | -52.2%  |
| 10  | 49.96   | 43.44    | -54.5% | -49.4%  |

**TimesFM:**
| k   | ID RMSE | OOD RMSE | ID Δ%  | OOD Δ%  |
| --- | ------- | -------- | ------ | ------- |
| 0   | 97.54   | 87.70    | —      | —       |
| 1   | 81.88   | 69.55    | -16.1% | -20.7%  |
| 3   | 57.50   | 47.83    | -41.0% | -45.5%  |
| 5   | 45.04   | 39.71    | -53.8% | -54.7%  |
| 10  | 50.61   | 43.53    | -48.1% | -50.4%  |

**TiRex:**
| k   | ID RMSE | OOD RMSE | ID Δ%  | OOD Δ%  |
| --- | ------- | -------- | ------ | ------- |
| 0   | 78.92   | 62.49    | —      | —       |
| 1   | 69.36   | 52.79    | -12.1% | -15.5%  |
| 3   | 50.05   | 39.71    | -36.6% | -36.5%  |
| 5   | 42.33   | 33.89    | -46.4% | -45.8%  |

**Chronos-Bolt-Tiny:**
| k   | ID RMSE | OOD RMSE | ID Δ%  | OOD Δ%  |
| --- | ------- | -------- | ------ | ------- |
| 0   | 111.65  | 90.69    | —      | —       |
| 1   | 69.15   | 57.05    | -38.1% | -37.1%  |
| 3   | 68.78   | 58.71    | -38.4% | -35.3%  |
| 5   | 71.99   | 62.68    | -35.5% | -30.9%  |
| 10  | 67.76   | 55.64    | -39.3% | -38.7%  |

</details>

**Key Findings**:
- 🎯 **TiRex achieves best absolute performance** at k=5: 42.33 ID RMSE, 33.89 OOD RMSE
- 📈 **k=5 is optimal** for large models (Chronos-2, TimesFM, TiRex) - k=10 shows performance degradation
- ⚡ **Chronos-Bolt-Tiny saturates at k=1** - larger model capacity needed to leverage more examples
- 🔄 **Strong OOD generalization** - few-shot improvements apply equally to in-distribution and out-of-distribution data
- 🚀 **Average improvement at k=5**: 48.3% ID RMSE reduction, 45.9% OOD RMSE reduction
- 💡 **Zero training required** - immediate deployment without finetuning, achieving ~60% of finetuning's gains

**Few-Shot vs Finetuning Comparison (Chronos-2):**
| Method             | ID RMSE | OOD RMSE | Training Required | Improvement vs Zero-Shot |
| ------------------ | ------- | -------- | ----------------- | ------------------------ |
| Zero-shot          | 84.86   | 60.78    | No                | —                        |
| Few-shot (k=5)     | 46.77   | 41.05    | No                | 45% ID / 32% OOD         |
| LoRA Finetuning    | 18.29   | 37.45    | Yes (3000 steps)  | 78% ID / 38% OOD         |
| Hyperparameter Opt | 15.41   | 34.05    | Yes (3000 steps)  | 82% ID / 44% OOD         |

**ICL Configuration**:
- Example format: Context-target pairs (80 context + 64 target timesteps)
- Example selection: Random sampling from training pool (246 traces)
- Normalization: Per-trace standardization (independent for each example)
- Test set protection: 6 ID test traces excluded from example pool
- Random seed: 42 (reproducible results)

### Finetuning Results

Performance of finetuned models on the same heat flux prediction task.

300 flux trace simulations were at our disposal. For the following experiments 49 traces were filtered out due to time series head and tail means not being between 1 and inf.
The remaining 251 samples were sub-sampled from 800 timesteps to 266 (every 3rd timestep) to resemble the benchmark time series as closely as possible.
For finetuning autogluon samples a custom validation set from the training data. This validation set consists of timeseries where the [:-prediction_length] tail is used for validation and hyperparameter tuning.
I created a routine to manually provide a validation set by sampling the training set timeseries in a stratified manner. This is achieved by assigning each time series to a bin depending on the time series mean.

> **Todo:** However, since also with this approach only the last prediction tail is used for validation and we want to predict time series from timestep 80 onwards I have to come up with a different solution.

To increase the amount of training data, I subsampled each flux trace three times with a distance of three timesteps.

| Model                               | Training Samples | Fine-tune Mode | Learning Rate | Steps    | Batch Size | Cross Learning | LoRA r | LoRA α | In-Distribution RMSE | In-Distribution SE | Out-of-Distribution RMSE | Out-of-Distribution SE | Date           |
| ----------------------------------- | ---------------- | -------------- | ------------- | -------- | ---------- | -------------- | ------ | ------ | -------------------- | ------------------ | ------------------------ | ---------------------- | -------------- |
| Chronos2 (Finetuned)                | 251              | LoRA           | 1e-4          | 3000     | 64         | No             | 16     | 32     | 18.29                | 3.54               | 37.45                    | 6.77                   | 2025-12-31     |
| **Chronos2 (Hyperparameter Tuned)** | **251**          | **LoRA**       | **1.77e-4**   | **3000** | **64**     | **Yes**        | **16** | **32** | **15.41**            | **2.55**           | **34.05**                | **9.87**               | **2026-01-01** |
| Chronos2 (Augmented Data)           | 753              | LoRA           | 1.77e-4       | 3000     | 64         | No             | 16     | 32     | 26.32                | 4.53               | 36.54                    | 10.56                  | 2026-01-01     |
| Chronos2 (Custom Val Set)           | 677              | LoRA           | 1.7e-4        | 5000     | 64         | No             | 16     | 32     | 24.45                | 6.94               | 38.17                    | 14.32                  | 2026-01-01     |
| Chronos2 (Custom Val Set)           | 677              | LoRA           | 5e-4          | 6000     | 72         | Yes            | 16     | 32     | 27.86                | 7.03               | 36.69                    | 17.63                  | 2026-01-01     |
| Chronos2 (Custom Val Set)           | 677              | LoRA           | 4.88e-4       | 4000     | 64         | Yes            | 32     | 64     | 31.36                | 7.18               | 39.30                    | 16.35                  | 2026-01-01     |

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
