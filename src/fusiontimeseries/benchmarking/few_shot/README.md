# Few-Shot In-Context Learning Benchmarks

This module implements few-shot in-context learning (ICL) benchmarks for foundation time-series models applied to fusion plasma flux prediction.

## Overview

**Goal**: Test whether pre-trained foundation models can improve predictions by learning from a small number of example traces at inference time, without any finetuning.

**Approach**:
- Provide k example traces (context + target pairs) to the model before the query
- Set example trace targets length between 64 (partial) and 187 (full trace) timesteps
- Random example selection from training pool
- Same evaluation metrics as zero-shot benchmarks for direct comparison

**Models Supported**:
- Chronos-2 (`amazon/chronos-2`)
- Chronos-Bolt-Tiny (`amazon/chronos-bolt-tiny`)
- TiRex (`NX-AI/TiRex`)
- TimesFM (`google/timesfm-2.5-200m-pytorch`)

## Quick Start

### 1. Run a Few-Shot Benchmark

```bash
# Open any notebook in Jupyter
cd src/fusiontimeseries/benchmarking/few_shot
uv run jupyter lab

# Open chronos2_fewshot_benchmark.ipynb
# Set K_SHOT = 3 (or 1, 5, 10)
# Set example_target_length=None for full traces or 64 for partial
# Run all cells
```

### 2. Results

Results are saved to:
- **JSON**: `results/few_shot/{timestamp}_{model}_k{k}_fewshot_results.json`
- **Plots**: `results/few_shot/plots/{timestamp}_{model}_k{k}/`

### 3. Compare to Zero-Shot

Each notebook prints zero-shot baseline results for comparison:

```
CHRONOS-2 FEW-SHOT (k=3) RESULTS
============================================================
ID RMSE:  XX.XX ± XX.XX
OOD RMSE: XX.XX ± XX.XX

Zero-shot baseline (for comparison):
ID RMSE:  84.86 ± 14.18
OOD RMSE: 60.78 ± 12.75
============================================================
```

## Architecture

### Module Structure

```
few_shot/
├── __init__.py                           # Module exports
├── few_shot_utils.py                     # Core utilities
├── chronos2_fewshot_benchmark.ipynb      # Benchmark Notebooks
├── chronos_bolt_tiny_fewshot_benchmark.ipynb
├── tirex_fewshot_benchmark.ipynb
├── timesfm_fewshot_benchmark.ipynb
└── README.md
```

### Core Components

**`few_shot_utils.py`** provides:

1. **`FewShotConfig`**: Configuration dataclass
   ```python
   config = FewShotConfig(
       model_slug="amazon/chronos-2",
       model_prediction_length=64,
       start_context_length=80,
       relevant_prediction_tail=80,
       k_shot=5,                      # Number of examples
       random_seed=42,                # For reproducibility
       example_target_length=None,    # RECOMMENDED: Use full traces
   )
   ```

2. **`FewShotExample`**: Example trace dataclass
   ```python
   example = FewShotExample(
       trace_id=42,
       trace=[...],          # Full trace (266 timesteps)
       context=[...],        # First 80 timesteps
       target=[...],         # Next 64 timesteps
   )
   ```

3. **`create_example_pool()`**: Create pool of training examples
   ```python
   test_ids = {8, 115, 131, 148, 235, 262}  # ID test set
   pool = create_example_pool(
       exclude_ids=test_ids,
       target_length=config.example_target_length  # Use config!
   )
   # Returns 245 examples (251 valid traces - 6 test IDs)
   # exclude_ids are RAW iteration ids, translated to pool positions via
   # operating_params.py (the old position-based exclusion leaked the test
   # traces' twins into the pool; fixed 2026-06-11)
   # Example format: context=80, target=187 (if target_length=None)
   ```

4. **`select_examples_random()`**: Randomly select k examples
   ```python
   examples = select_examples_random(pool, k=3, seed=42)
   # Reproducible selection
   ```

5. **`format_context_target_pairs()`**: Format for ICL
   ```python
   icl_context = format_context_target_pairs(examples, query_context)
   # Returns: [ex1_ctx(80), ex1_tgt(64), ex2_ctx(80), ex2_tgt(64), ..., query(80)]
   ```

## ICL Format

**Context-Target Pairs**: Each example consists of:
- **Context**: First 80 timesteps (linear phase)
- **Target**: Configurable via `example_target_length` parameter
  - **Recommended**: `None` (full trace = 187 timesteps)
  - Historical: `64` (partial, limited performance)

**ICL Input Structure**:
```
[example_1_context, example_1_target,
 example_2_context, example_2_target,
 ...
 example_k_context, example_k_target,
 query_context]
```

This provides full demonstrations of context → target predictions before the query.

**Example Context Lengths (Recommended: Full Traces)**:
- k=1:  1 × (80 + 187) + 80 = **347 timesteps**
- k=3:  3 × (80 + 187) + 80 = **881 timesteps**
- k=5:  5 × (80 + 187) + 80 = **1,415 timesteps**
- k=10: 10 × (80 + 187) + 80 = **2,750 timesteps**

Some of these may exceed model context windows; adjust k, example_target_length accordingly.

## Normalization Strategy

**Per-Trace Normalization** (matches zero-shot baseline):
- Each example normalized independently on its own context (first 80 timesteps)
- Query normalized independently on its own context
- Examples and query remain on different scales
- Consistent with zero-shot evaluation

```python
# Example normalization
ex_scaler = StandardScaler()
normed_ctx = ex_scaler.fit_transform(ex.context.reshape(-1, 1))
normed_tgt = ex_scaler.transform(ex.target.reshape(-1, 1))

# Query normalization (separate scaler)
query_scaler = StandardScaler()
normed_query = query_scaler.fit_transform(query.reshape(-1, 1))
```

## Autoregressive Prediction

**Strategy**: Include examples at every autoregressive step
1. **First step**: `[examples + query_context(80)]` → predict 64 steps
2. **Second step**: `[examples + query_context(80) + prediction(64)]` → predict 64 more
3. **Third step**: `[examples + query_context(80) + prediction(128)]` → predict 64 more
4. Continue until 266 timesteps

## Test Set Protection

**Critical**: Example pool MUST NOT include test set traces

**In-Distribution Test IDs**:
- 8, 115, 131, 148, 235, 262

**Out-of-Distribution Test IDs**:
- 0, 1, 2, 3, 4 (with `_realpotens` suffix)

**Verification**:
```python
from fusiontimeseries.benchmarking.few_shot.operating_params import ID_TEST_RAW_IDS

pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS))
pool_ids = {ex.trace_id for ex in pool}
assert not (pool_ids & ID_TEST_RAW_IDS), "Test set leakage!"
assert len(pool) == 245
```

The id-based assert alone is NOT sufficient — it passed while the old
position-based exclusion was leaking all six test twins. The module
self-test (`python -m ...few_shot.few_shot_utils`) additionally checks BY
VALUE that no pool trace matches any benchmark trace under any subsample
phase.

## Evaluation Metrics

**Same as zero-shot benchmarks**:
- RMSE with standard error (Delta method)
- Evaluated on mean of last 80 timesteps
- Separate metrics for ID and OOD

## Experiment Workflow

### 1. Test Different k Values

```python
# In any notebook, change K_SHOT and rerun
K_SHOT = 1   # 1-shot
K_SHOT = 3   # 3-shot (default)
K_SHOT = 5   # 5-shot
K_SHOT = 10  # 10-shot
```

### 2. Collect Results

```bash
# Results are saved with k in filename
results/few_shot/
├── 20260105_180000_amazon_chronos-2_k1_fewshot_results.json
├── 20260105_180500_amazon_chronos-2_k3_fewshot_results.json
├── 20260105_181000_amazon_chronos-2_k5_fewshot_results.json
└── 20260105_181500_amazon_chronos-2_k10_fewshot_results.json
```

### 3. Compare Performance

**Hypotheses to Test**:
- **H1**: k-shot improves over zero-shot (lower RMSE)
- **H2**: Larger k improves performance (up to a point)
- **H3**: Few-shot helps more on ID than OOD
- **H4**: Full trace examples outperform short targets

## Results

### Zero-Shot Baseline (k=0)

| Model | ID RMSE ± SE | OOD RMSE ± SE |
|-------|--------------|---------------|
| TiRex | 78.92 ± 13.43 | 62.49 ± 18.12 |
| Chronos-2 | 109.91 ± 14.80 | 85.86 ± 23.55 |
| TimesFM | 97.54 ± 13.18 | 87.70 ± 26.96 |
| Chronos-bolt-tiny | 111.65 ± 13.92 | 90.69 ± 21.80 |

---

### Few-Shot Results: Short Targets (target_length=64, 144 total)

#### Chronos-2 (Short Targets)

| k | ID RMSE ± SE | OOD RMSE ± SE |
|---|--------------|---------------|
| 0 | 109.91 ± 14.80 | 85.86 ± 23.55 |
| 1 | 75.69 ± 8.22 | 61.60 ± 15.84 |
| 3 | 53.60 ± 9.58 | 46.71 ± 17.00 |
| 5 | **46.77 ± 8.98** | **41.05 ± 17.06** |
| 10 | 49.96 ± 9.19 | 43.44 ± 17.19 |

Best: k=5 → 46.77 / 41.05 RMSE

#### TiRex (Short Targets)

| k | ID RMSE ± SE | OOD RMSE ± SE |
|---|--------------|---------------|
| 0 | 78.92 ± 13.43 | 62.49 ± 18.12 |
| 1 | 69.36 ± 11.24 | 52.79 ± 17.24 |
| 3 | 50.05 ± 8.22 | 39.71 ± 17.60 |
| 5 | **42.33 ± 8.95** | **33.89 ± 14.03** |

Best: k=5 → 42.33 / 33.89 RMSE

#### TimesFM (Short Targets)

| k | ID RMSE ± SE | OOD RMSE ± SE |
|---|--------------|---------------|
| 0 | 97.54 ± 13.18 | 87.70 ± 26.96 |
| 1 | 81.88 ± 10.37 | 69.55 ± 17.60 |
| 3 | 57.50 ± 9.18 | 47.83 ± 16.76 |
| 5 | **45.04 ± 8.87** | **39.71 ± 17.04** |
| 10 | 50.61 ± 8.94 | 43.53 ± 17.34 |

Best: k=5 → 45.04 / 39.71 RMSE

#### Chronos-bolt-tiny (Short Targets)

| k | ID RMSE ± SE | OOD RMSE ± SE |
|---|--------------|---------------|
| 0 | 111.65 ± 13.92 | 90.69 ± 21.80 |
| 1 | 69.15 ± 10.77 | 57.05 ± 16.68 |
| 3 | 68.78 ± 10.16 | 58.71 ± 16.97 |
| 5 | 72.00 ± 9.95 | 62.68 ± 17.07 |
| 10 | 67.76 ± 10.41 | 55.64 ± 16.89 |
| 15 | **56.64 ± 9.50** | **48.00 ± 17.16** |

Best: k=15 → 56.64 / 48.00 RMSE

---

### Few-Shot Results: Full Traces (target_length=None, 267 total)

#### Chronos-2 (Full Traces)

| k | ID RMSE ± SE | OOD RMSE ± SE |
|---|--------------|---------------|
| 0 | 109.91 ± 14.80 | 85.86 ± 23.55 |
| 1 | 49.04 ± 9.40 | 40.03 ± 17.25 |
| 3 | 43.00 ± 8.62 | 37.76 ± 17.06 |
| 5 | **38.98 ± 8.18** | **35.56 ± 16.04** |
| 10 | 43.40 ± 9.23 | 38.96 ± 17.11 |

Best: k=5 → 38.98 / 35.56 RMSE

#### TiRex (Full Traces)

| k | ID RMSE ± SE | OOD RMSE ± SE |
|---|--------------|---------------|
| 0 | 78.92 ± 13.43 | 62.49 ± 18.12 |
| 1 | 43.61 ± 9.43 | 38.60 ± 16.66 |
| 3 | 41.22 ± 8.72 | 37.60 ± 16.07 |
| 5 | **30.52 ± 8.07** | **33.81 ± 12.56** |
| 10 | 35.14 ± 8.42 | 35.19 ± 14.99 |

Best: k=5 → 30.52 / 33.81 RMSE

#### TimesFM (Full Traces)

| k | ID RMSE ± SE | OOD RMSE ± SE |
|---|--------------|---------------|
| 0 | 97.54 ± 13.18 | 87.70 ± 26.96 |
| 1 | 45.16 ± 9.29 | 38.89 ± 16.62 |
| 3 | 47.54 ± 8.70 | 41.48 ± 16.96 |
| 5 | **34.93 ± 8.15** | **34.00 ± 14.54** |
| 10 | 35.13 ± 8.09 | 34.25 ± 15.08 |

Best: k=5 → 34.93 / 34.00 RMSE

#### Chronos-bolt-tiny (Full Traces)

| k | ID RMSE ± SE | OOD RMSE ± SE |
|---|--------------|---------------|
| 0 | 111.65 ± 13.92 | 90.69 ± 21.80 |
| 1 | 50.86 ± 9.59 | 44.77 ± 17.58 |
| 3 | 29.64 ± 8.55 | **33.18 ± 11.84** |
| 5 | 34.14 ± 9.97 | 33.56 ± 12.47 |
| 10 | **24.09 ± 7.77** | 35.50 ± 9.18 |

Best ID: k=10 → 24.09 RMSE
Best OOD: k=3 → 33.18 RMSE

---

### Summary: Best Results Across All Experiments

| Model | Target Length | k | ID RMSE ± SE | OOD RMSE ± SE |
|-------|---------------|---|--------------|---------------|
| **TiRex** | Full (267) | 5 | **30.52 ± 8.07** | **33.81 ± 12.56** |
| **Chronos-bolt-tiny** | Full (267) | 10 | **24.09 ± 7.77** | 35.50 ± 9.18 |
| **Chronos-bolt-tiny** | Full (267) | 3 | 29.64 ± 8.55 | **33.18 ± 11.84** |
| **TimesFM** | Full (267) | 5 | **34.93 ± 8.15** | **34.00 ± 14.54** |
| **Chronos-2** | Full (267) | 5 | **38.98 ± 8.18** | **35.56 ± 16.04** |

Overall best ID RMSE: Chronos-bolt-tiny k=10 full → **24.09 ± 7.77**
Overall best OOD RMSE: TiRex k=5 full → **33.81 ± 12.56**
Overall best balanced: TiRex k=5 full → **30.52 / 33.81 RMSE**

---

### Comparison to Finetuning

| Approach | Training Required | ID RMSE | OOD RMSE |
|----------|-------------------|---------|----------|
| Zero-Shot | None | 78.92 | 62.49 |
| Few-Shot (k=5 full) | None | 30.52 | 33.81 |
| LoRA Finetuning | 251 traces | 18.29 | 37.45 |

Model: TiRex for zero-shot/few-shot, Chronos-2 for finetuning (from CLAUDE.md)
