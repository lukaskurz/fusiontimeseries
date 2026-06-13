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

**OOD is mostly finetuning's story** ($67.94 \to 34.10$ at $k=0$); *shape*
retrieval does not improve it (mmr $+1.9$–$3.1$), but *level-aware* retrieval
does (Part A, below). The **512-window clamp is a context-composition effect,
not window-length mismatch**: it beats full-window in all 8 mmr cells (under
the clamp $k{=}5 \equiv k{=}10$ bit-identically — only the last example's tail
+ query survive) yet *destroys* the oracle ceiling ($9.39 \to 19.57$ ID) — a
crude tail-selector that drops the wrong-level mass legit retrieval inevitably
includes. The shipped step-200 checkpoint beats the final step-4000 weights
everywhere (overtraining degrades ICL most).

**Level-aware retrieval on the finetuned model (Part A).** Phase 6 used only
the shape-matching `mmr_euclid`; the
[`level_matching`](../results/fewshot/level_matching.md) follow-up later showed
that matching the absolute context *level* (`|mean(ctx)−mean(query)|`, the
signal the tail metric scores) beats shape-matching dramatically OOD. Re-running
the grid with `ctx_level` and a level-aware MMR hybrid (`mmr_level`: level
relevance + shape diversity) on the *finetuned* model reproduces the
level-vs-shape split cleanly: **`ctx_level` significantly improves ft OOD**
($34.10 \to 27.42$ at $k{=}5$ mean, $p_\text{boot}=0.000$; $\to 24.58$ under the
512 clamp — the best legitimate ft OOD), while `mmr_euclid` leaves OOD *at or
above* ft $k{=}0$ (36.00) yet still wins ID (18.62 vs `ctx_level`'s 32.32).
`mmr_level` lands between on both axes (OOD 29.57 / ID 34.64 at $k{=}5$ mean),
dominating neither — its shape-diversity penalty re-admits the wrong-level mass
that hurts OOD. The takeaway is the model-free verdict carried onto the
finetuned model: **shape retrieval for ID, level retrieval for OOD — no single
80-step-context retriever wins both.** This is consistent with the
level-calibration mechanism (below): the OOD gain is exactly a level-bias
reduction the shape distance z-scores away.

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

## In-context finetuning (Phase 9)

Phase 6 found the finetuned model's in-context ability is *inherited* from base
pretraining — `train_bilinear.py` finetunes on single subsampled traces, never
on demonstrations. **In-context finetuning (ICF)** trains the same BilinearLoRA
recipe ON multi-example ICL concatenations
([`Chronos2ICLDataset`](../../src/fusiontimeseries/finetuning/chronos2/icl_dataset.py),
window 2048, $k\in\{1,3,5\}$ per sample, raw concatenation = level-clean,
query-only conditioning) so the model can *learn* to use demonstrations. Two
checkpoints isolate the mechanism:
[`train_bilinear_icl.py`](../../src/fusiontimeseries/finetuning/chronos2/train_bilinear_icl.py)
trains a **level** model (demos retrieved by context level, train≡test
`ctx_level`) and a **random** control (demos sampled at random). Full results:
[`icl_finetuning_table.md`](../results/fewshot/icl_finetuning_table.md) (+
`icl_finetuning_kcurve.png`).

**ICF makes the model demonstration-dependent, and the model genuinely learns
to USE level-matched demos — but only the oracle reveals it.** Both ICF
checkpoints collapse at $k=0$ (level 59.76 / random 46.52 ID vs the v6
single-trace model's 22.20) and gain massively from ICL ($-29$ ID,
$p\le0.012$): trained always-with-demos, they are demo-driven where v6 was not.
The **random control is the decisive probe**: given perfectly level-matched
demonstrations (the cheating oracle), ICF-level dominates ICF-random — **OOD
7.49 vs 61.82** ($\Delta-54.3$, $p_\text{boot}=0.000$), ID 22.20 vs 32.80
($p=0.042$) — and the random model gains essentially nothing from oracle demos
(it learned to *ignore* demo level, so it cannot exploit level-matched ones).
This is the cleanest separation in the study.

**But ICF does not beat the inherited ICL ability under realistic retrieval.**
With `ctx_level` retrieval ICF-level $\approx$ the v6 ft model (ID $-0.27$ n.s.,
OOD $+1.9$ n.s.) and the project's best legitimate ID (15.63, v6 ft `mmr_euclid`
@ the 512 window) is unbeaten. ICF moved the *ceiling* (level-oracle OOD 7.49
beats v6's oracle 10.89, $p=0.000$), not the realized number — the same
80-step-context information limit as Phase 7. A sharp **ID/OOD personality
split** appears under realistic retrieval: ICF-random is ID-optimised
(`ctx_level` $k{=}10$ 16.35 ID but catastrophic 57 OOD — a level-blind
amplitude trick) while ICF-level is balanced (~30/30) because it relies on level
matching, which transfers to OOD. The *training*-demo distribution sets which
axis the model optimises, mirroring the eval-time shape-for-ID / level-for-OOD
split one layer up. **Bottom line: the bottleneck was never in-context capacity
(pretrained or ICF-trained) — it is retrieval; closing it needs side
information, not more in-context training.**

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
  `run_finetuned_grid.py`, `analyze_finetuned.py` (Phase 6, + Part A
  level-aware retrieval cells); `run_mechanism_dump.py`, `analyze_mechanism.py`
  (Phase 7); `run_icl_finetuned.py`, `analyze_icl_finetuned.py` (Phase 9 ICF);
  training `finetuning/chronos2/train_bilinear.py` (single-trace) and
  `train_bilinear_icl.py` + `icl_dataset.py` (ICF).
- Results:
  [`finetuned_icl_table.md`](../results/fewshot/finetuned_icl_table.md),
  [`mechanism_table.md`](../results/fewshot/mechanism_table.md),
  [`icl_finetuning_table.md`](../results/fewshot/icl_finetuning_table.md) +
  figures.
- Self-tests: `python -m fusiontimeseries.benchmarking.few_shot.finetuned`;
  `... analyze_mechanism --self-test`.
- Method context for the conditioning: [`BilinearLoRA.md`](BilinearLoRA.md).
- Narrative context: [`few_shot_icl.md`](few_shot_icl.md).
