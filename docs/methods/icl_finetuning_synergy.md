# ICL × Finetuning Synergy and the Level-Calibration Mechanism

Two questions, one answer. **Does adaptation stack** — does retrieval-ICL add
anything on top of a finetuned model (Phase 6)? And **where do the gains come
from** — does few-shot calibrate amplitude or genuinely improve dynamics
(Phase 7)? Both reduce to a single mechanism: every gain on this benchmark is
**level calibration**.

## Motivation

**RAF** ([2411.08249](https://arxiv.org/pdf/2411.08249)) reports retrieval and
finetuning are *synergistic*. If retrieval-ICL adds anything on top of
Severin's operating-parameter-conditioned BilinearLoRA, that is the strongest
joint result the thesis has; if it does not (or hurts — the finetuned model
never saw concatenated splices), that is a citable negative. Either way it
bridges both halves of the project. Phase 7 then asks *why* the ladder works,
framed against the context-parroting literature: TSFMs forecast chaotic systems
by copying context motifs, fail pointwise quickly, yet preserve invariant
statistics — and our tail-mean metric is exactly such an invariant statistic.

## Method

**The 2×2.** $\{\text{base}, \text{finetuned}\} \times \{k=0, \text{best-}k\}$,
everything else protocol-identical (shared scaling, fixed pool, both decodings,
one grid run, identical example sets hard-asserted). The finetuned model is
self-trained with Severin's exact recipe
([`train_bilinear.py`](../../src/fusiontimeseries/finetuning/chronos2/train_bilinear.py));
the grid is checkpoint-agnostic (sha256 in every JSON). Finetuned forwards are
conditioned on the *query's* raw parameters $[\hat{s}, q, R/L_T, R/L_n]$ via
`ConditionRegistry`. A `win512` variant clamps the ICL stream to the model's
512-step training window.

**Mechanism decomposition (Phase 7).** Per trace, the tail MSE splits exactly:

$$
\text{MSE}_{\text{tail}}
= \underbrace{\bar{b}^2}_{\text{level bias}^2}
+ \underbrace{\overline{e_{\text{fluc}}^2}}_{\text{fluctuation error}^2},
\qquad
\bar{b} = \overline{\hat{y}}^{\text{tail}} - \overline{y}^{\text{tail}}
$$

(the JSONs' `error` field *is* $\bar b$). Thirteen headline cells were re-run
with full forecast trajectories dumped through a new `forecast_callback` harness
hook (`run_mechanism_dump.py`), every per-trace scalar reproducing its v5/v6
value **bit-exactly** (143/143). Further diagnostics: a tracking horizon on a
smoothed normalized error, ACF / correlation-time $\tau_c$ / flatline stats,
and the oracle-gap feature hunt (`analyze_mechanism.py`).

## Findings

Phase 6:
[`docs/results/fewshot/finetuned_icl_table.md`](../results/fewshot/finetuned_icl_table.md)
(+ `finetuned_synergy.png`, regenerated `adaptation_ladder.png`).
Phase 7:
[`docs/results/fewshot/mechanism_table.md`](../results/fewshot/mechanism_table.md)
(+ `mechanism_*.png`, `forecast_grid_*.png`).

**Adaptation stacks on ID — through retrieval quality.** Finetuning alone
($k=0$, 22.20 ID, mean decoding) already beats the best base ICL cell (27.06);
retrieval-ICL takes it to 18.62 (`mmr_euclid` $k=5$), and clamping to the 512
training window to **15.63 ID — the project's best legitimate number**. With
$n=6$ traces the marginal ICL gain is *not* individually significant (CI
$[-16.4, +16.8]$); the direction is consistent across all mmr cells and both
windows. (Phase 8's `[0::3]` reconciliation shows this best-config gain is
phase-sensitive — see [evaluation reconciliation](../results/fewshot/evaluation_reconciliation.md)
and [few_shot_icl.md](few_shot_icl.md); the robust facts — ft $k=0$ ≈ 22 ID,
phase-stable — survive.)

**ICL capacity survives finetuning — the bottleneck is retrieval.** The
cheating oracle stacks *significantly* on the finetuned model (9.39 ID
$p=0.043$; 10.89 OOD $p=0.002$ vs ft $k=0$), and the ft model exploits oracle
examples better than the base does ($-12.2$ ID, $p=0.032$). Conversely
**random examples destroy the finetuned advantage** (ft ≈ base at random
$k=10$, $+0.03$ ID). The op_knn-on-the-finetuned-model probe (the one selector
uniquely motivated there, since the model *is* conditioned on those params)
scores ≈ random (39.6 vs 40.0 ID) — Phase-2's "params don't beat context"
verdict generalizes even to the conditioned model.

**OOD is finetuning's story alone** ($67.94 \to 34.10$ at $k=0$; no legit ICL
config improves it). The **512-window clamp is a context-composition effect,
not window-length mismatch**: it beats full-window in all 8 mmr cells (under
the clamp $k{=}5 \equiv k{=}10$ bit-identically — only the last example's tail
+ query survive) yet *destroys* the oracle ceiling ($9.39 \to 19.57$ ID) — a
crude tail-selector that drops the wrong-level mass legit retrieval inevitably
includes. The shipped step-200 checkpoint beats the final step-4000 weights
everywhere (overtraining degrades ICL most).

**The mechanism is level calibration — all of it.** In all 16
config-vs-anchor comparisons the level term absorbs ~100% of the change
($\Delta \bar b^2 / \Delta\text{MSE} = 0.87$–$1.12$), while the fluctuation
error sits at $0.68$–$0.89\times$ the chaos floor $\sigma\sqrt{2}$ for *every*
config — nothing phase-tracks the turbulence. Since $\sqrt{\overline{\bar b^2}}$
*is* the benchmark RMSE, "better RMSE" and "better level calibration" are the
same statement — which is why shared scaling (Phase 3) was the unlock. **No
genuine horizon extension**: median tracking horizons ~8–22 steps vs the
truth's own correlation time 8–9; every rollout under-disperses (std ratio
$\le 0.6$, frequent flatlines) and over-smooths. **The oracle gap is an
information limit of the 80-step context**: the pool holds a near-exact level
twin for every query (pool-min level distance $\le 0.9$, median 0.23), but
every retrieval distance z-scores the context, erasing the level signal — the
oracle's picks rank ~98/245 under context distance. Closing the gap needs side
information, not a better 80-step distance.

**Metric audit.** The chronos2 finetuning notebooks score `mean(x[:-80])`
(including the 80 copied context steps); the honest tail rescore of the *same*
forecasts gives ID 17.51 / **OOD 40.64** (vs 15.72 / 6.03 on the notebook
metric) — the dramatic chronos2 finetuning OOD numbers are largely this
artifact. See the [evaluation reconciliation](../results/fewshot/evaluation_reconciliation.md).

## Relationship to prior work

The synergy claim is **RAF**
([2411.08249](https://arxiv.org/pdf/2411.08249)); we qualitatively support it on
ID but show the gain is bottlenecked by retrieval, not the model's in-context
capacity. The mechanism connects to the chaotic-systems line: **context
parroting** (Zhang & Gilpin, [2505.11349](https://arxiv.org/abs/2505.11349)) and
**zero-shot forecasting of chaotic systems**
([2409.15771](https://arxiv.org/abs/2409.15771)) — TSFMs copy context motifs and
preserve invariant statistics after pointwise failure; our tail mean is such an
invariant. The collapse to a level (no dispersion, over-smoothing) is the
**transformer-ICL collapse-to-the-mean** behaviour analyzed in
[2510.09776](https://arxiv.org/abs/2510.09776).

## Code and results

- Implementation:
  [`benchmarking/few_shot/finetuned.py`](../../src/fusiontimeseries/benchmarking/few_shot/finetuned.py),
  `run_finetuned_grid.py`, `analyze_finetuned.py` (Phase 6);
  `run_mechanism_dump.py`, `analyze_mechanism.py` (Phase 7);
  training `finetuning/chronos2/train_bilinear.py`.
- Results:
  [`finetuned_icl_table.md`](../results/fewshot/finetuned_icl_table.md),
  [`mechanism_table.md`](../results/fewshot/mechanism_table.md) + figures.
- Self-tests: `python -m fusiontimeseries.benchmarking.few_shot.finetuned`;
  `... analyze_mechanism --self-test`.
- Method context for the conditioning: [`BilinearLoRA.md`](BilinearLoRA.md).
- Narrative context: [`few_shot_icl.md`](few_shot_icl.md).
