# Training-Free Operating-Parameter Conditioning via Covariates

Can the four operating parameters $(q, \hat{s}, R/L_T, R/L_n)$ be injected into
the forecast **without any training**, through Chronos-2's zero-shot covariate
support? This document describes the conditioning scheme in
[`benchmarking/few_shot/covariates.py`](../../src/fusiontimeseries/benchmarking/few_shot/covariates.py)
(Phase 4) and the clean **negative** result, plus the structural reason it must
fail.

## Motivation

This rung completes the *adaptation ladder* for the joint thesis story —
zero-shot → ICL → ICL + OP covariates → finetuned OP conditioning (Severin's
BilinearLoRA) — with monotonically increasing adaptation cost. Chronos-2 is the
only benchmarked model with zero-shot covariate support: `predict()` accepts
dict tasks with `past_covariates` / `future_covariates`, attended through group
attention with no finetuning.

## Method

**The load-bearing fact.** Chronos-2 (like Chronos-Bolt) instance-normalizes
**every variate row independently** with
$\text{loc} = \operatorname{nanmean}(\text{row})$,
$\text{scale} = \sqrt{\operatorname{nanmean}((\text{row}-\text{loc})^2)}$
(arcsinh after standardization — monotone, so the invariances carry over). This
norm is invariant under any positive-affine map $r \mapsto a r + b$ ($a>0$).
Two consequences:

1. A **constant** covariate row has its *value erased exactly*: in exact
   arithmetic $(c-c)/\varepsilon = 0$; in float32 the row mean rounds within
   1 ulp, leaving a constant all-$0/\pm 1$ "tri-state" row whose sign is FP
   rounding noise, not physics (verified bit-identical for two engineered
   values sharing tri-states). A constant channel therefore cannot *encode* the
   operating point — it injects value-uncorrelated perturbation.
2. Raw values and $[0,1]$-normalized values produce **identical** post-norm
   rows; only *within-row contrast* survives.

So static parameters must act through within-row contrast — encoded as **step
functions over the flat-concat ICL stream**: each example's normalized
parameters held constant over its `[ctx, tgt]` segment, the query's over its
(growing) segment, and `future_covariates` = the query's value over the
prediction window.

**Controls.** A *permuted-params* control permutes example parameter values
among the selected examples (deterministic per (query, set)) — without it a
$+\text{cov}$ change is not attributable to parameter *information* vs the mere
*presence* of extra covariate rows. A *group+cov* variant (constant query
channels on the Phase-3 group task) makes the "static covariates are inert in
group mode" claim an explicit table row.

## Findings

Grid in `results/few_shot_v4_covariates/`, analysis in
[`docs/results/fewshot/covariates_table.md`](../results/fewshot/covariates_table.md)
(+ `adaptation_ladder.png`, `covariates_kcurves.png`,
`covariates_contrast_scatter.png`).

- **Training-free conditioning does not work, and the controls show why.**
  $+\text{cov}$ is statistically indistinguishable from the permuted-params
  control at every comparable cell (random $k=10$: 48.48 vs 47.90 ID) — the
  channels contribute *presence*, not parameter *information*.
- **The presence acts as a level-homogenizing perturbation**, pulling every
  strategy toward ≈ 47 ID: it *helps* weak anchors (zero-shot + constant
  channels $109.91 \to 81.02$ ID, while provably carrying no information) and
  *destroys* strong ones (oracle $k=10$: $23.31 \to 47.28$ ID, CI $[8.1,
  35.0]$).
- Group-mode covariates are structurally inert as predicted, slightly harmful
  in practice.

The conditioning signal this metric needs — absolute level from absolute
parameters — cannot survive the per-row instance norm without training, which
is precisely the gap finetuned conditioning closes (see
[icl_finetuning_synergy.md](icl_finetuning_synergy.md)).

## Relationship to prior work

The intended mechanism is **FiLM**-style feature modulation
([1709.07871](https://arxiv.org/abs/1709.07871)) / adaLN as in **DiT**
([2212.09748](https://arxiv.org/abs/2212.09748)) — but those *learn* the
conditioning transform, whereas zero-shot covariates rely on the model's frozen
instance-norm, which is provably level-blind. The Severin BilinearLoRA family
([BilinearLoRA.md](BilinearLoRA.md), [RSSBilinearLoRA.md](RSSBilinearLoRA.md))
is exactly the trained FiLM-in-rank-space conditioning that succeeds where this
training-free attempt cannot. **Chronos-2**
([2510.15821](https://arxiv.org/abs/2510.15821)) supplies the covariate API.

## Code and results

- Implementation:
  [`benchmarking/few_shot/covariates.py`](../../src/fusiontimeseries/benchmarking/few_shot/covariates.py)
  (`build_op_channels`, `make_chronos2_covariate_forecast_fn`,
  `make_chronos2_group_covariate_forecast_fn`); grid `run_covariates_grid.py`;
  analysis `analyze_covariates.py`.
- Results:
  [`docs/results/fewshot/covariates_table.md`](../results/fewshot/covariates_table.md)
  + `adaptation_ladder.png`.
- Self-test: `python -m fusiontimeseries.benchmarking.few_shot.covariates`.
- Narrative context: [`few_shot_icl.md`](few_shot_icl.md).
