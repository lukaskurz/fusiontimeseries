# Point-Forecast Decoding and Ensembling

The benchmark scores the RMSE of a time-averaged tail of a positive,
right-skewed flux. The RMSE-optimal point forecast of such a quantity is the
conditional **mean** — yet every model wrapper decoded the **median**. This
document describes the decoding knob in
[`benchmarking/few_shot/run_decoding_grid.py`](../../src/fusiontimeseries/benchmarking/few_shot/run_decoding_grid.py)
(Phase 5) and the per-model decision it produced, plus the ensembling verdicts.

## Motivation

For a target $Y$ and prediction $\hat{y}$, $\mathbb{E}[(Y-\hat{y})^2]$ is
minimized at $\hat{y} = \mathbb{E}[Y]$. For a right-skewed $Y$ the median P50
sits *below* the mean, so median decoding systematically **under-estimates**
the saturation level — exactly the error the metric punishes. The fix is a
one-knob change through the existing harness.

## Method

`decode_point_forecast(quantiles, point_stat)` decodes a point from a
9-quantile forecast:

- **`median`** — $q_{0.5}$ (index $n_q // 2$), the frozen Phase-1..4 path.
- **`mean`** — the decile average $\tfrac{1}{9}\sum_{j} q_{0.1..0.9}$. Caveat:
  it *truncates* the tails beyond $q_{0.1}/q_{0.9}$, so it is a "decile-average
  mean" that is itself biased *low* on a heavy right tail — an under-correction,
  not the exact conditional mean.
- **`meanhead`** (TimesFM only) — its native mean output head (index 0 of the
  $[\text{mean}, q_{0.1..0.9}]$ output), the unbiased cross-check.

Two wiring facts the implementation rests on: **TiRex has no native mean** (its
`forecast()` "mean" return is a relabeled median — `# median as mean` selects
$q_{0.5}$ by index; confirmed bit-identical at runtime), and the decoded point
**feeds back through the autoregressive rollout**, so decoding changes whole
trajectories, not just the final read-out.

**Ensembling** (analyzer-side, since the tail mean is linear in the forecast):
*seed ensembling* averages the 20 random-example-set forecasts per trace before
scoring; *cross-model ensembling* averages best-config forecasts across models.

## Findings

Grid in `results/few_shot_v5_decoding/` (36 cells; median cells bit-reproduce
their Phase-3/4 twins), analysis in
[`docs/results/fewshot/decoding_table.md`](../results/fewshot/decoding_table.md)
(+ `decoding_effect.png`).

- **Mean decoding helps where level calibration is poor — and the worse the
  config, the more it helps.** Mean improves ID RMSE in 14 of 16 (model,
  config) cells: dramatically at zero-shot anchors (Chronos-2 $109.91 \to
  89.51$ ID, $-20.4$, significant both splits), clearly for random selection
  (Chronos-2 $-5.2$ ID), marginally at the already-calibrated best configs
  (Bolt $-0.65$). It produces the new best legitimate training-free cell:
  **22.63 ID** (Bolt `mmr_euclid` shared $k=10$ + mean).
- **The decision is per-model.** Adopt **mean** for Chronos-2 (uniform gains,
  significant at zero-shot/random/oracle — and the default for the finetuned
  Chronos-2 phase), Bolt and TiRex (small free wins, never significantly
  worse). **Keep median for TimesFM**: at its best config the decile mean is
  significantly *worse* ($+1.05$ ID / $+1.57$ OOD) and its native mean head
  worse still ($+2.01$ / $+3.46$) — since meanhead $\ge$ decile-mean $\ge$
  median there, the failure is the mean *statistic* interacting with TimesFM's
  wide right tail under ICL, not the decile-truncation bias.
- **Seed ensembling works; cross-model ensembling does not.** Averaging the 20
  random-set forecasts beats per-seed scoring for every model/decoding ($-1.7$
  to $-5.7$ ID, all bootstrap $p \le 0.001$) but its best cell (Bolt mean
  34.27 ID) still loses to plain retrieval — a fallback, not a replacement.
  Cross-model ensembles (all pairs + the 4-model average) never beat the best
  single model ID: the models' tail-level errors are too correlated.

This is the first direct evidence that the few-shot metric is *level*-bound, a
theme [Phase 7](icl_finetuning_synergy.md) makes exact.

## Relationship to prior work

The conditional-mean argument is textbook decision theory; the bagging result
follows **Modi & Pan** (ensembling TSFM forecasts,
[2508.16641](https://arxiv.org/abs/2508.16641)), which reports up to 54%
variance reduction from averaging TSFM forecasts — we confirm the variance
reduction is real and significant but is dominated by retrieval here, because
the residual error is shared level bias (correlated across models), not
independent noise.

## Code and results

- Implementation: decoding knob in
  [`benchmarking/few_shot/rerun_ksweep.py`](../../src/fusiontimeseries/benchmarking/few_shot/rerun_ksweep.py)
  (`decode_point_forecast`, `POINT_STATS`); grid `run_decoding_grid.py`;
  analysis `analyze_decoding.py` (seed/model ensembling).
- Results:
  [`docs/results/fewshot/decoding_table.md`](../results/fewshot/decoding_table.md)
  + `decoding_effect.png`.
- Narrative context: [`few_shot_icl.md`](few_shot_icl.md).
