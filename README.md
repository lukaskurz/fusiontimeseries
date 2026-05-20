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
