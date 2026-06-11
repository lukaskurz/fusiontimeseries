# 🧰 Installation & Environment

These instructions were tested on Windows (PowerShell). Adjust paths/commands as needed for macOS/Linux.

## Prerequisites

Install [uv](https://docs.astral.sh/uv/#installation)

## Installation Steps

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

## Notes

- `nbstripout` will clear outputs from `.ipynb` files before they are committed.
- Ruff is configured to attempt automatic fixes (`--fix`) where appropriate.

## Quick Test

Run the [TiRex Playbook](../playground/tirex-playbook.ipynb) from top to bottom to confirm a working environment.

## Platform Support

This repository and the included notebook have been validated on Windows (PowerShell). If you run into platform-specific issues on macOS/Linux, please file an issue or open a PR with repro steps.
