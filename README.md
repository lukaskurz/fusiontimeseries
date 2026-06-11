# ⚛️ Flux Time Series Prediction in Tokamak Reactors

![Python 3.13](https://img.shields.io/badge/python-3.13-blue?style=flat-square&logo=python&logoColor=white)
![UV](https://img.shields.io/pypi/v/uv?label=uv&style=flat-square&logo=pypi&logoColor=white)


Welcome to the Fusion Time Series playground — example code and notebooks for experimenting with flux time-series forecasting models and surrounding tooling.

## 🧰 Installation

See the [Installation Guide](docs/installation.md) for detailed setup instructions.

## 📚 Documentation

```
docs/
├── methods/
│   ├── BilinearLoRA.md
│   ├── OSSBilinearLoRA.md
│   └── RSSBilinearLoRA.md
├── poster/
├── report/
│   └── 0126-progress-report.md
├── results/
│   ├── finetuning/
│   │   ├── chronos2/
│   │   └── timesfm/
│   └── zeroshot/
└── installation.md
```

- **[methods/](docs/methods/)** - LoRA adaptation techniques documentation
- **[poster/](docs/poster/)** - Poster presentations
- **[report/](docs/report/)** - Progress reports and documentation
- **[results/](docs/results/)** - Experimental results
  - **[finetuning/](docs/results/finetuning/)** - Fine-tuning experiment results
  - **[zeroshot/](docs/results/zeroshot/)** - Zero-shot model results
- **[installation.md](docs/installation.md)** - Installation and setup guide

## 📊 Results

### Zero-Shot Results

| Base Model               | ID $\bar{Q}$ (↓)  | OOD $\bar{Q}$     | Inference Time [s]  |
| ------------------------ | ----------------- | ----------------- | ------------------- |
| google/timesfm-2.0-500m  | 156.17 ± 67.31    | 98.61 ± 23.55     | 5.65 ± 5.82e-2      |
| amazon/chronos-bolt-tiny | 110.15 ± 14.08    | 92.89 ± 21.16     | **0.030 ± 1.08e-3** |
| amazon/chronos2          | 107.09 ± 15.74    | 87.47 ± 22.08     | 0.073 ± 1.72e-3     |
| google/timesfm-2.5-200m  | 104.23 ± 14.87    | 87.20 ± 25.05     | 0.231 ± 7.23e-3     |
| NX-AI/TiRex              | **79.49 ± 14.38** | **64.03 ± 19.53** | 1.95 ± 1.63e-2      |

Zero-shot performance across five time-series foundation models.

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


| Base Model                     | Finetuning Type  | ID $\bar{Q}$ (↓) | OOD $\bar{Q}$   | Trainable Params (%) | Trainable Params (#Mio.) | Inference Time [s] |
| ------------------------------ | ---------------- | ---------------- | --------------- | -------------------- | ------------------------ | ------------------ |
| google/timesfm-2.0-500m        | Full Finetuning* | 20.67 ± 7.43     | 12.01 ± 3.21    | 100.0                | 498.8                    | 0.091 ± 1.65e-3    |
| google/timesfm-2.0-500m        | BilinearLoRA     | 20.15 ± 7.79     | 7.11 ± 1.32     | 1.22                 | 6.2                      | 0.245 ± 2.17e-3    |
| google/timesfm-2.0-500m        | OSSBilinearLoRA  | 19.24 ± 7.87     | 7.74 ± 2.08     | 28.91                | 202.8                    | 0.291 ± 3.85e-3    |
| GyroSwin-1B [[1]](#references) | -                | 18.35 ± 1.56     | 26.43 ± 9.49    | 100.0                | 1000.0                   | 2.849**            |
| google/timesfm-2.0-500m        | RSSBilinearLoRA  | 18.03 ± 6.81     | 7.86 ± 2.20     | 1.39                 | 7.0                      | 0.304 ± 2.50e-3    |
| google/timesfm-2.0-500m        | LoRA*            | 17.76 ± 8.05     | 16.07 ± 4.18    | 1.02                 | 5.1                      | 0.081 ± 1.51e-3    |
| amazon/chronos2                | LoRA*            | 16.73 ± 6.67     | 5.08 ± 1.22     | 1.0                  | 1.2                      | 0.067 ± 2.95e-2    |
| amazon/chronos2                | RSSBilinearLoRA  | 16.33 ± 5.39     | 5.65 ± 2.03     | 1.86                 | 2.3                      | 0.170 ± 6.26e-3    |
| amazon/chronos2                | OSSBilinearLoRA  | 16.11 ± 6.18     | **3.19 ± 0.73** | 25.0                 | 39.8                     | 0.159 ± 4.59e-3    |
| amazon/chronos2                | Full Finetuning* | 15.50 ± 4.47     | 4.76 ± 0.89     | 100.0                | 119.5                    | 0.050 ± 7.07e-4    |
| amazon/chronos2                | BilinearLoRA     | **13.83 ± 4.18** | 4.86 ± 0.68     | 1.54                 | 1.9                      | 0.136 ± 8.64e-4    |

- Comparison of finetuned performance across base models and GyroSwin
- For average heat flux $\bar{Q}$ we report RMSE of time-averaged predictions after an autoregressive rollout
- Time-series models are trained and benchmarked on a NVIDIA RTX 4070 16GB Ti Super
- (*) No operating parameter conditioning
- (**) To compare the inference speed to GyroSwin we use the reported 15.4ms forward pass inference speed and multiply by the number of rollout steps (185). The large speed gap can be mainly attributed to the fact that time-series models forecast 64 timesteps in one forward-pass. GyroSwin was benchmarked on a NVIDIA H100 80GB HBM3.

## References

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


Ruff check

# repos:
#   - repo: https://github.com/astral-sh/ruff-pre-commit
#     rev: v0.15.0
#     hooks:
#       - id: ruff-check
#         args: ["--fix"]
#       - id: ruff-format

#   - repo: https://github.com/kynan/nbstripout
#     rev: 0.9.0
#     hooks:
#       - id: nbstripout
#         args:
#           ["--extra-keys=metadata.celltoolbar cell.metadata.heading_collapsed"]

#   - repo: https://github.com/pre-commit/pre-commit-hooks
#     rev: v6.0.0
#     hooks:
#       - id: end-of-file-fixer
#       - id: trailing-whitespace
