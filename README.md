# ⚛️ Flux Time Series Prediction in Tokamak Reactors

![Python 3.13](https://img.shields.io/badge/python-3.13-blue?style=flat-square&logo=python&logoColor=white)
![UV](https://img.shields.io/pypi/v/uv?label=uv&style=flat-square&logo=pypi&logoColor=white)

Can pre-trained time-series foundation models predict turbulent heat flux in fusion reactors? This project benchmarks zero-shot, few-shot in-context learning, and finetuning performance of state-of-the-art time-series foundation models (TiRex, Chronos, TimesFM) on heat flux traces from gyrokinetic plasma turbulence simulations.

The work is split between two contributors:

- **Zero-shot benchmarking & finetuning** — Severin Bergsmann ([@sbergsmann](https://github.com/sbergsmann))
- **Few-shot in-context learning** — Lukas Kurz ([@lukaskurz](https://github.com/lukaskurz))

## 🔬 Background: The GyroSwin Paper

This project builds on **GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations** ([Paischer et al., 2025](https://arxiv.org/abs/2510.07314), see [References](#references)). GyroSwin is a neural surrogate for the nonlinear gyrokinetic equations governing plasma turbulence: it evolves the full 5D distribution function $f(k_x, k_y, s, v_\parallel, \mu)$ over time and predicts 3D electrostatic potential fields and the scalar heat flux $\bar{Q}$, replacing prohibitively expensive numerical simulations (three orders of magnitude speedup over the GKW solver).

**The data**: 255 nonlinear simulations generated with the GKW gyrokinetic code (adiabatic electron approximation), varying four operating parameters: safety factor $q$, magnetic shear $\hat{s}$, ion temperature gradient $R/L_T$, and density gradient $R/L_n$.

**Our angle**: instead of modelling the 5D state, we treat the heat flux $\bar{Q}(t)$ as a plain 1D time series and ask how far modern time-series foundation models get — with no physics, no operating parameters, and (for zero-/few-shot) no training at all.

## 📐 Evaluation Protocol

We follow the GyroSwin paper's evaluation exactly, so our numbers are directly comparable to theirs:

- **Test sets**: 6 in-distribution (ID) simulations (inside the convex hull of the training parameters, but unseen) and 5 out-of-distribution (OOD) simulations (outside the convex hull)
- **Metric**: RMSE of the **time-averaged heat flux** $\bar{Q}$ over the final 80 timesteps, after an autoregressive rollout
- **Setup**: traces are subsampled every 3rd step (800 → 266 timesteps); models see the first 80 timesteps (linear phase) as context and forecast autoregressively in steps of 64
- Per-trace z-score normalization, seed 42

## 📊 Results

### 1. Baselines from the GyroSwin Paper

What we are competing against — heat flux RMSE as reported in the paper (Table 2, same 6 ID + 5 OOD simulations, same metric):

| Method                  | Type                          | ID $\bar{Q}$ (↓) | OOD $\bar{Q}$ (↓) |
| ----------------------- | ----------------------------- | ---------------- | ----------------- |
| QL / QuaLiKiz           | Reduced-order quasilinear     | 89.53 ± 11.76    | 95.22 ± 21.57     |
| GPR                     | 0D surrogate (op. params)     | 43.82 ± 10.84    | 59.28 ± 17.55     |
| MLP                     | 0D surrogate (op. params)     | 50.50 ± 10.79    | 61.98 ± 18.41     |
| FNO                     | Neural PDE surrogate          | 119.88 ± 13.15   | 124.96 ± 23.27    |
| PointNet                | Neural PDE surrogate          | 119.93 ± 13.15   | 125.05 ± 23.29    |
| Transolver              | Neural PDE surrogate          | 119.93 ± 13.15   | 125.05 ± 23.28    |
| ViT                     | Neural PDE surrogate          | 119.63 ± 13.13   | 125.13 ± 23.29    |
| GyroSwin (48 sims)      | 5D surrogate                  | 67.68 ± 10.28    | 70.48 ± 17.21     |
| **GyroSwin-1B (Large)** | 5D surrogate (241 sims)       | **18.35 ± 1.56** | **26.43 ± 9.49**  |

The neural PDE surrogates (FNO, PointNet, Transolver, ViT) all collapse to nearly identical, poor heat flux predictions — capturing $\bar{Q}$ from the 5D state is hard. The strongest baseline is the full GyroSwin-1B model trained on 241 simulations.

### 2. Zero-Shot Results

*by Severin Bergsmann*

Pre-trained time-series foundation models applied out of the box — no training, no fusion data, no operating parameters:

| Base Model               | ID $\bar{Q}$ (↓)  | OOD $\bar{Q}$ (↓) | Inference Time [s]  |
| ------------------------ | ----------------- | ----------------- | ------------------- |
| google/timesfm-2.0-500m  | 156.17 ± 67.31    | 98.61 ± 23.55     | 5.65 ± 5.82e-2      |
| amazon/chronos-bolt-tiny | 110.15 ± 14.08    | 92.89 ± 21.16     | **0.030 ± 1.08e-3** |
| amazon/chronos2          | 107.09 ± 15.74    | 87.47 ± 22.08     | 0.073 ± 1.72e-3     |
| google/timesfm-2.5-200m  | 104.23 ± 14.87    | 87.20 ± 25.05     | 0.231 ± 7.23e-3     |
| NX-AI/TiRex              | **79.49 ± 14.38** | **64.03 ± 19.53** | 1.95 ± 1.63e-2      |

**Takeaway**: with zero exposure to fusion data, TiRex (79.49 ID / 64.03 OOD) already beats the paper's quasilinear QuaLiKiz baseline (89.53 / 95.22) and all neural PDE surrogates, and matches the 48-simulation GyroSwin on ID while clearly beating it on OOD. Forecast plots per model are in [docs/results/zeroshot/](docs/results/zeroshot/).

### 3. Few-Shot In-Context Learning Results

*by Lukas Kurz*

The same pre-trained models, but provided with $k$ example traces (80-timestep context + 64-timestep target pairs, randomly sampled from the 246-trace training pool, test traces excluded) at inference time — still **no finetuning, no gradient updates**.

**Best configuration per model:**

| Model                           | k   | ID RMSE (↓)    | OOD RMSE (↓)    | Improvement vs Zero-Shot |
| ------------------------------- | --- | -------------- | --------------- | ------------------------ |
| **NX-AI/TiRex**                 | 5   | **42.33 ± 8.95** | **33.89 ± 14.03** | **46.4% ID / 45.8% OOD** |
| google/timesfm-2.5-200m-pytorch | 5   | 45.04 ± 8.87   | 39.71 ± 17.04   | 53.8% ID / 54.7% OOD     |
| amazon/chronos-2                | 5   | 46.77 ± 8.98   | 41.05 ± 17.06   | 57.4% ID / 52.2% OOD     |
| amazon/chronos-bolt-tiny        | 1   | 69.15 ± 10.77  | 57.05 ± 16.68   | 38.1% ID / 37.1% OOD     |

<details>
<summary><b>Full k-shot learning curves</b> (k = 0, 1, 3, 5, 10)</summary>

**Chronos-2:**
| k   | ID RMSE | OOD RMSE | ID Δ%  | OOD Δ% |
| --- | ------- | -------- | ------ | ------ |
| 0   | 109.91  | 85.86    | —      | —      |
| 1   | 75.69   | 61.60    | -31.1% | -28.3% |
| 3   | 53.60   | 46.71    | -51.2% | -45.6% |
| 5   | 46.77   | 41.05    | -57.4% | -52.2% |
| 10  | 49.96   | 43.44    | -54.5% | -49.4% |

**TimesFM-2.5:**
| k   | ID RMSE | OOD RMSE | ID Δ%  | OOD Δ% |
| --- | ------- | -------- | ------ | ------ |
| 0   | 97.54   | 87.70    | —      | —      |
| 1   | 81.88   | 69.55    | -16.1% | -20.7% |
| 3   | 57.50   | 47.83    | -41.0% | -45.5% |
| 5   | 45.04   | 39.71    | -53.8% | -54.7% |
| 10  | 50.61   | 43.53    | -48.1% | -50.4% |

**TiRex:**
| k   | ID RMSE | OOD RMSE | ID Δ%  | OOD Δ% |
| --- | ------- | -------- | ------ | ------ |
| 0   | 78.92   | 62.49    | —      | —      |
| 1   | 69.36   | 52.79    | -12.1% | -15.5% |
| 3   | 50.05   | 39.71    | -36.6% | -36.5% |
| 5   | 42.33   | 33.89    | -46.4% | -45.8% |

**Chronos-Bolt-Tiny:**
| k   | ID RMSE | OOD RMSE | ID Δ%  | OOD Δ% |
| --- | ------- | -------- | ------ | ------ |
| 0   | 111.65  | 90.69    | —      | —      |
| 1   | 69.15   | 57.05    | -38.1% | -37.1% |
| 3   | 68.78   | 58.71    | -38.4% | -35.3% |
| 5   | 71.99   | 62.68    | -35.5% | -30.9% |
| 10  | 67.76   | 55.64    | -39.3% | -38.7% |

</details>

**Key findings:**

- 🎯 **TiRex at k=5 achieves the best training-free result**: 42.33 ID / 33.89 OOD RMSE — better than every paper baseline except the full GyroSwin-1B, at zero training cost
- 📈 **k=5 is the sweet spot** for the larger models (TiRex, TimesFM, Chronos-2); k=10 degrades performance
- ⚡ **Chronos-Bolt-Tiny saturates at k=1** — more examples need more model capacity
- 🔄 **Improvements transfer to OOD**: few-shot gains are nearly identical in- and out-of-distribution (~48% ID / ~46% OOD average reduction at k=5)
- 💡 In-context examples recover a large share of finetuning's gains (see below) **without any gradient updates**

### 4. Finetuning Results

*by Severin Bergsmann*

Models finetuned on the training traces with operating-parameter conditioning, including the bilinear LoRA variants documented in [docs/methods/](docs/methods/):

| Base Model                     | Finetuning Type  | ID $\bar{Q}$ (↓) | OOD $\bar{Q}$ (↓) | Trainable Params (%) | Trainable Params (#Mio.) | Inference Time [s] |
| ------------------------------ | ---------------- | ---------------- | ----------------- | -------------------- | ------------------------ | ------------------ |
| google/timesfm-2.0-500m        | Full Finetuning* | 20.67 ± 7.43     | 12.01 ± 3.21      | 100.0                | 498.8                    | 0.091 ± 1.65e-3    |
| google/timesfm-2.0-500m        | BilinearLoRA     | 20.15 ± 7.79     | 7.11 ± 1.32       | 1.22                 | 6.2                      | 0.245 ± 2.17e-3    |
| google/timesfm-2.0-500m        | OSSBilinearLoRA  | 19.24 ± 7.87     | 7.74 ± 2.08       | 28.91                | 202.8                    | 0.291 ± 3.85e-3    |
| GyroSwin-1B [[1]](#references) | -                | 18.35 ± 1.56     | 26.43 ± 9.49      | 100.0                | 1000.0                   | 2.849**            |
| google/timesfm-2.0-500m        | RSSBilinearLoRA  | 18.03 ± 6.81     | 7.86 ± 2.20       | 1.39                 | 7.0                      | 0.304 ± 2.50e-3    |
| google/timesfm-2.0-500m        | LoRA*            | 17.76 ± 8.05     | 16.07 ± 4.18      | 1.02                 | 5.1                      | 0.081 ± 1.51e-3    |
| amazon/chronos2                | LoRA*            | 16.73 ± 6.67     | 5.08 ± 1.22       | 1.0                  | 1.2                      | 0.067 ± 2.95e-2    |
| amazon/chronos2                | RSSBilinearLoRA  | 16.33 ± 5.39     | 5.65 ± 2.03       | 1.86                 | 2.3                      | 0.170 ± 6.26e-3    |
| amazon/chronos2                | OSSBilinearLoRA  | 16.11 ± 6.18     | **3.19 ± 0.73**   | 25.0                 | 39.8                     | 0.159 ± 4.59e-3    |
| amazon/chronos2                | Full Finetuning* | 15.50 ± 4.47     | 4.76 ± 0.89       | 100.0                | 119.5                    | 0.050 ± 7.07e-4    |
| amazon/chronos2                | BilinearLoRA     | **13.83 ± 4.18** | 4.86 ± 0.68       | 1.54                 | 1.9                      | 0.136 ± 8.64e-4    |

- (*) No operating parameter conditioning
- (**) GyroSwin inference time estimated from the reported 15.4 ms forward pass × 185 rollout steps (benchmarked on an NVIDIA H100 80GB); time-series models forecast 64 timesteps per forward pass and were benchmarked on an NVIDIA RTX 4070 Ti Super 16GB

**Takeaway**: finetuned Chronos-2 (13.83 ID / 4.86 OOD with BilinearLoRA, ~1.9M trainable parameters) outperforms the 1B-parameter GyroSwin on both test sets — most dramatically OOD — while being orders of magnitude cheaper to train and run. Forecast plots are in [docs/results/finetuning/](docs/results/finetuning/).

## 🧰 Installation

See the [Installation Guide](docs/installation.md) for detailed setup instructions.

## 📚 Documentation

```
docs/
├── methods/           # BilinearLoRA, OSSBilinearLoRA, RSSBilinearLoRA write-ups
├── poster/            # Poster presentations
├── report/            # Progress reports
├── results/
│   ├── finetuning/    # Finetuning forecast plots (chronos2/, timesfm/)
│   └── zeroshot/      # Zero-shot forecast plots
└── installation.md
```

## References

[1] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations

```bibtex
@misc{paischer2025gyroswin5dsurrogatesgyrokinetic,
      title={GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations},
      author={Fabian Paischer and Gianluca Galletti and William Hornsby and Paul Setinek and Lorenzo Zanisi and Naomi Carey and Stanislas Pamela and Johannes Brandstetter},
      year={2025},
      eprint={2510.07314},
      archivePrefix={arXiv},
      primaryClass={physics.plasm-ph},
      url={https://arxiv.org/abs/2510.07314},
}
```
