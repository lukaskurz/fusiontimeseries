# Few-Shot In-Context Learning Benchmarks

This module implements few-shot in-context learning (ICL) benchmarks for foundation time-series models applied to fusion plasma flux prediction.

## Overview

**Goal**: Test whether pre-trained foundation models can improve predictions by learning from a small number of example traces at inference time, without any finetuning.

**Approach**:
- Provide k example traces (context + target pairs) to the model before the query
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
├── chronos2_fewshot_benchmark.ipynb      # Chronos-2 benchmark
├── chronos_bolt_tiny_fewshot_benchmark.ipynb
├── tirex_fewshot_benchmark.ipynb
├── timesfm_fewshot_benchmark.ipynb
└── README.md                             # This file
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
       k_shot=3,                # Number of examples
       random_seed=42,          # For reproducibility
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
   pool = create_example_pool(exclude_ids=test_ids)
   # Returns ~246 examples (251 valid traces - 6 test IDs)
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
- **Target**: Next 64 timesteps (prediction window)

**ICL Input Structure**:
```
[example_1_context, example_1_target,
 example_2_context, example_2_target,
 ...
 example_k_context, example_k_target,
 query_context]
```

This provides full demonstrations of context → target predictions before the query.

**Example Context Lengths**:
- k=1:  1 × (80 + 64) + 80 = **224 timesteps**
- k=3:  3 × (80 + 64) + 80 = **512 timesteps**
- k=5:  5 × (80 + 64) + 80 = **800 timesteps**
- k=10: 10 × (80 + 64) + 80 = **1520 timesteps**

All values are well within model context limits (Chronos-2: 8192 tokens).

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

**Rationale**: Consistent ICL signal throughout prediction (vs. examples only in first step)

## Model-Specific Details

### Chronos-2
- **Input shape**: `[1, 1, context_length]` (batch, channel, timesteps)
- **Output**: Quantile forecasts `[1, n_quantiles, pred_length]`
- **ICL support**: Native (group attention mechanism)

### Chronos-Bolt-Tiny
- **Input shape**: `[1, context_length]` (batch, timesteps)
- **Output**: Quantile forecasts `[1, n_quantiles, pred_length]`
- **ICL support**: Similar to Chronos-2 (smaller model)

### TiRex
- **Input shape**: `[context_length]` (1D tensor)
- **Output**: Quantile forecasts `[1, pred_length, n_quantiles]`
- **ICL support**: xLSTM state tracking

### TimesFM
- **Input**: Numpy array `[context_length]`
- **Output**: Point forecasts `[1, pred_length]` (no quantiles)
- **ICL support**: Base model (may benefit from TimesFM-ICF variant)

## Test Set Protection

**Critical**: Example pool MUST NOT include test set traces

**In-Distribution Test IDs**:
- 8, 115, 131, 148, 235, 262

**Out-of-Distribution Test IDs**:
- 0, 1, 2, 3, 4 (with `_realpotens` suffix)

**Verification**:
```python
test_ids = {8, 115, 131, 148, 235, 262}
pool = create_example_pool(exclude_ids=test_ids)
pool_ids = {ex.trace_id for ex in pool}
assert not (pool_ids & test_ids), "Test set leakage!"
```

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
- **H4**: Models with better ICL support show larger gains

**Analysis**:
```python
import json
import pandas as pd

# Load results for different k
results = {}
for k in [0, 1, 3, 5, 10]:  # 0 = zero-shot baseline
    with open(f"results/few_shot/*_k{k}_*.json") as f:
        results[k] = json.load(f)

# Compare RMSE
df = pd.DataFrame({
    'k': [0, 1, 3, 5, 10],
    'ID_RMSE': [results[k]['in_distribution']['rmse'] for k in [0, 1, 3, 5, 10]],
    'OOD_RMSE': [results[k]['out_of_distribution']['rmse'] for k in [0, 1, 3, 5, 10]],
})
print(df)

# Plot improvement
import matplotlib.pyplot as plt
plt.plot(df['k'], df['ID_RMSE'], marker='o', label='ID')
plt.plot(df['k'], df['OOD_RMSE'], marker='o', label='OOD')
plt.xlabel('k (number of examples)')
plt.ylabel('RMSE')
plt.title('Few-Shot Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
```

## Zero-Shot Baselines

For comparison, here are the zero-shot results:

| Model | ID RMSE | OOD RMSE |
|-------|---------|----------|
| Chronos-2 | 84.86 ± 14.18 | 60.78 ± 12.75 |
| Chronos-bolt-tiny | 87.78 ± 13.76 | 68.02 ± 13.00 |
| TiRex | 63.91 ± 13.62 | 44.79 ± 7.92 |
| TimesFM | 82.79 ± 11.69 | 62.78 ± 14.51 |

**Target**: Few-shot RMSE ≤ Zero-shot RMSE

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce k or use a smaller model
```python
K_SHOT = 1  # Use fewer examples
# Or use Chronos-bolt-tiny instead of Chronos-2
```

### Issue: Test Set Leakage Warning
**Solution**: Verify test IDs are excluded
```python
test_ids = {8, 115, 131, 148, 235, 262}
pool = create_example_pool(exclude_ids=test_ids)
# Check pool_ids don't overlap with test_ids
```

### Issue: Results Not Improving
**Possible causes**:
1. Model doesn't support ICL well → Try Chronos-2 (best ICL support)
2. Examples not diverse enough → Consider implementing nearest-neighbor selection
3. Normalization mismatch → Verify per-trace normalization

## Future Extensions

**Out of scope for initial implementation, but possible enhancements**:

1. **Advanced Example Selection**:
   - Nearest-neighbor (select similar traces)
   - Diverse sampling (k-means clusters)
   - Stratified sampling (span flux ranges)

2. **Alternative Normalization**:
   - Joint normalization (examples + query together)
   - Global normalization (all examples together)

3. **Alternative ICL Formats**:
   - Context-only (cheaper, no targets)
   - Full traces (266 timesteps each)

4. **Model Variants**:
   - TimesFM-ICF (specialized for ICL)
   - Different separator tokens

## References

- **Chronos-2 Paper**: "Chronos-2: From univariate to universal forecasting" (arXiv:2510.15821)
- **TiRex Paper**: "TiRex: Zero-Shot Forecasting with Enhanced In-Context Learning" (NeurIPS 2025)
- **TimesFM-ICF**: "Time series foundation models can be few-shot learners" (Google Research)

## Contact

For questions or issues, refer to the main project README or contact the maintainer.

**Last Updated**: 2026-01-05
