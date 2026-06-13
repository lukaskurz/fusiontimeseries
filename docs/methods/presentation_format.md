# Example Presentation Format: Shared vs Per-Example Scaling

Phase 2 showed that *which* examples are retrieved cannot help while each is
z-scored independently. This document describes the presentation fixes in
[`benchmarking/few_shot/presentation.py`](../../src/fusiontimeseries/benchmarking/few_shot/presentation.py)
(Phase 3) and the headline result: **shared scaling** is what lets retrieval
work, and it is the single most important knob on the few-shot side.

## Motivation

Two known weaknesses of naive flat-concatenation ICL:

1. **Level erasure (Phase-2 diagnosis).** Each demonstration is z-scored with
   its own mean/std, so its absolute saturation level is normalized away before
   the model sees it. The metric scores *level*, so the matched signal never
   arrives.
2. **Splice discontinuities.** [TimesFM-ICF](https://arxiv.org/pdf/2410.24087)
   (ICML 2025) showed concatenating series without separators looks like one
   jagged stream and degrades ICL. Our $k=10$ stream contains 10 fake
   discontinuities.

The harness rollout stays frozen; everything new lives in `presentation.py`.

## Method

**Per-example vs shared scaling.** Let the query context fit a scaler
$\mathcal{S}_q$ (mean $\mu_q$, std $\sigma_q$ over the first 80 steps).

- *Per-example* (the Phase-1/2 default, reproduced bit-for-bit): example $i$ is
  transformed by its own $\mathcal{S}_i$. A model that copies an example's tail
  level $\ell_i$ in normalized space denormalizes through $\mathcal{S}_q$ back
  to $\mu_q + \sigma_q (\ell_i - \mu_i)/\sigma_i$ — the example's *shape*, mapped
  to the query's scale, with its absolute level destroyed.
- *Shared*: example contexts **and** targets are transformed by the **one**
  query-fit scaler $\mathcal{S}_q$. Now copying $\ell_i$ in normalized space
  round-trips to the example's *true* raw level $\ell_i$ — the level survives
  the normalize → predict → denormalize cycle (self-test T2 verifies this
  exactly). Fitting on the query (not on examples+query) keeps the frame
  independent of the selected set and reduces to zero-shot at $k=0$.

**Chronos-2 group ICL.** Examples passed as `past_covariates` rows of one dict
task (attended via GroupSelfAttention) instead of a spliced concatenation —
the principled fix for the splice artifact. But Chronos-2 instance-norms each
covariate row independently, so group mode cannot restore absolute level either
(see [operating_param_covariates.md](operating_param_covariates.md) for the
affine-invariance argument). Rows are NaN-left-padded to a common length.

**Ordering.** `make_ordered_select_fn` reorders picks (similar-last /
similar-first / shuffled, the last deterministic per (seed, query)).

**Truncation.** `truncate_example` keeps each example up to its overshoot peak
$+$ margin, applied *after* selection (truncating the pool would corrupt
rankings) — motivated by fixed context budgets (TimesFM 2048, TiRex training
length).

## Findings

Staged grid in `results/few_shot_v3_presentation/`, table + significance tests
in
[`docs/results/fewshot/presentation_table.md`](../results/fewshot/presentation_table.md)
(figures: `presentation_norm_ablation.png`, `presentation_group_vs_concat.png`).

- **Shared scaling confirms the Phase-2 diagnosis and inverts the picture.**
  The cheating oracle drops from ≈ random to its best everywhere (TimesFM
  15.99 ID / 8.10 OOD at $k=10$; TiRex 30.65 / 18.01) and now beats shared-norm
  random with bootstrap-significant margins on **all four models, both splits**
  — headroom that simply did not exist under per-example scoring. The flip
  side: *random* under shared scaling gets significantly **worse** ID for 3 of
  4 models (wrong levels now transfer), so shared scaling is only useful *with*
  retrieval. Best legitimate training-free ID improves $30.65 \to$ **23.28**
  (Bolt, `mmr_euclid` shared $k=10$).
- **Group ICL is much worse than flat concat** on identical example sets
  (random $k=5$: 79.1 vs 42.1 ID; still 68–72 vs 38–42 at $k=20$): covariate
  rows barely act as demonstrations, and per-row instance norm means group
  cannot restore level. Group ICL and shared scaling fix *different* problems;
  only the level one matters here.
- **Ordering is a non-factor** ($\pm 1$–4 RMSE, no consistent direction).
- **Truncation backfires under shared scaling** (TiRex `ctx_euclid` $k=10$:
  42.9 truncated vs 30.8 full) — it cuts off exactly the saturation tail the
  level mechanism copies from.

## Relationship to prior work

The splice-artifact hypothesis is **TimesFM-ICF**
([2410.24087](https://arxiv.org/pdf/2410.24087)); our result refines it — for a
*level*-dominated metric the discontinuity is second-order and **normalization
of absolute scale** is the dominant factor, which the in-context-finetuning
literature does not isolate. **Chronos-2**
([2510.15821](https://arxiv.org/abs/2510.15821)) supplies the native group-ICL
alternative we test; its per-row instance norm is precisely why group mode is
level-blind here.

## Code and results

- Implementation:
  [`benchmarking/few_shot/presentation.py`](../../src/fusiontimeseries/benchmarking/few_shot/presentation.py)
  (`make_concat_forecast_fn`, `make_chronos2_group_forecast_fn`,
  `make_ordered_select_fn`, `make_truncated_select_fn`); grid
  `run_presentation_grid.py`; analysis `analyze_presentation.py`.
- Results:
  [`docs/results/fewshot/presentation_table.md`](../results/fewshot/presentation_table.md)
  + `presentation_norm_ablation.png`, `presentation_group_vs_concat.png`.
- Self-test: `python -m fusiontimeseries.benchmarking.few_shot.presentation`.
- Narrative context: [`few_shot_icl.md`](few_shot_icl.md).
