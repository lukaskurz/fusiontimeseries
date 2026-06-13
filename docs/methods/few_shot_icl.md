# Few-Shot In-Context Learning for Heat-Flux Prediction

*The umbrella narrative for the few-shot side of the project (Lukas). This is
the document the thesis draws from; the per-phase method docs below carry the
detail and the result tables carry the numbers.*

## Overview

We ask how far a **pre-trained time-series foundation model** (TiRex,
Chronos-2, Chronos-Bolt, TimesFM) can predict the turbulent heat flux
$\bar{Q}(t)$ of a gyrokinetic plasma simulation when given $k$ example traces
at inference time — no gradient updates. The model sees the first 80 timesteps
(the linear growth phase) of a held-out simulation, plus $k$ demonstration
traces, and rolls forward autoregressively; the benchmark scores the RMSE of
the time-averaged saturation level $\bar{Q}$ over the final 80 steps, on 6
in-distribution and 5 out-of-distribution simulations, exactly as the GyroSwin
paper. This is a *level*-dominated metric on a chaotic, right-skewed,
bursty signal — a fact that turns out to explain almost everything.

The work proceeds as a chain of ablations, each isolating one axis. The single
thread running through all of them: **the few-shot gain is saturation-level
calibration, and every design choice is judged by whether it lets the right
level reach the model.**

1. **Example selection** ([example_selection.md](example_selection.md), Phase 2)
   — operating-parameter kNN, context-similarity (Euclidean / DTW / growth
   rate), MMR, and a cheating oracle. *Finding:* retrieval does **not** beat
   random ID, and neither does the oracle — because each example is z-scored
   independently, erasing the level the metric and the oracle match on. (op_knn
   does not beat context similarity; small consistent OOD gains.)
2. **Presentation format** ([presentation_format.md](presentation_format.md),
   Phase 3) — the fix Phase 2 demanded. **Shared scaling** (one query-fit
   scaler for examples *and* query) lets absolute level survive the
   normalize→predict→denormalize round trip; now the oracle works on all four
   models and the best legitimate ID drops $30.65 \to 23.28$. Chronos-2 group
   ICL is much worse (per-row instance norm is level-blind); ordering is a
   non-factor; truncation backfires.
3. **Operating-parameter covariates**
   ([operating_param_covariates.md](operating_param_covariates.md), Phase 4) —
   conditioning on $(q,\hat s,R/L_T,R/L_n)$ through Chronos-2's zero-shot
   covariates. A **clean negative**: the per-row instance norm erases a constant
   channel's value exactly (provable, verified bit-identical), so step-encoded
   channels carry *presence*, not information — homogenizing every strategy
   toward ≈47 ID. Absolute-level conditioning needs *training*.
4. **Decoding and ensembling**
   ([decoding_and_ensembling.md](decoding_and_ensembling.md), Phase 5) — the
   RMSE-optimal point forecast of a right-skewed quantity is the conditional
   **mean**, not the median every wrapper decoded. Mean decoding helps 14/16
   cells (most where calibration is worst); the decision is per-model (mean for
   Chronos-2/Bolt/TiRex, median for TimesFM). Seed ensembling helps but loses to
   retrieval; cross-model ensembling never wins.
5. **ICL × finetuning and the mechanism**
   ([icl_finetuning_synergy.md](icl_finetuning_synergy.md), Phases 6–7) —
   adaptation **stacks on ID through retrieval quality** (ft $k=0$ 22.20 already
   beats the best base ICL; ft+ICL $\to$ 18.62, $\to$ 15.63 at the 512 training
   window — the best legit number). The mechanism analysis proves **all gains
   are level calibration** (the level term absorbs ~100% of every MSE change;
   nothing phase-tracks the turbulence; no genuine horizon extension), and the
   oracle gap is an **information limit of the 80-step context**, not a
   distance-metric failure.

The adaptation ladder that ties it together —
zero-shot $\to$ ICL $\to$ (+ training-free covariates) $\to$ finetuned $\to$
finetuned + ICL — is the bridge deliverable connecting the few-shot and
finetuning halves of the project.

## Related work

**In-context fine-tuning / ICL for TSFMs.** TimesFM-ICF
([2410.24087](https://arxiv.org/pdf/2410.24087)) shows concatenated series
without separators degrade ICL and proposes a learnable separator + continued
pretraining; our Phase-3 result refines this — for a level-dominated metric the
splice discontinuity is second-order and **normalization of absolute scale** is
the dominant factor. Chronos-2
([2510.15821](https://arxiv.org/abs/2510.15821)) provides native group/covariate
ICL, whose per-row instance norm is precisely why group mode and zero-shot
covariates are level-blind here.

**Retrieval-augmented forecasting.** RAF
([2411.08249](https://arxiv.org/pdf/2411.08249)), TS-RAG
([2503.07649](https://arxiv.org/pdf/2503.07649)), TimeRAF
([2412.20810](https://arxiv.org/pdf/2412.20810)), and RAFT
([2511.05859](https://arxiv.org/pdf/2511.05859)) retrieve similar history to
condition forecasts — mostly *with* training. We isolate the training-free
selection axis (Phase 2: retrieval ≈ random until the level can transfer) and
the retrieval × finetuning interaction (Phase 6: RAF-style synergy is supported
on ID but bottlenecked by retrieval, not in-context capacity).

**ICL example selection.** Liu et al., *What Makes Good In-Context Examples for
GPT-3?* ([2101.06804](https://arxiv.org/abs/2101.06804)) — good demonstrations
matter, but (our refinement) only when the model can use the matched feature;
here the format gates the absolute level.

**Conditioning.** FiLM ([1709.07871](https://arxiv.org/abs/1709.07871)) and DiT
adaLN ([2212.09748](https://arxiv.org/abs/2212.09748)) motivate
operating-parameter conditioning; the trained BilinearLoRA family
([BilinearLoRA.md](BilinearLoRA.md), [RSSBilinearLoRA.md](RSSBilinearLoRA.md),
[OSSBilinearLoRA.md](OSSBilinearLoRA.md)) realizes it in rank space and succeeds
where the training-free covariate attempt cannot.

## Discussion

**Context parroting and invariant statistics.** The chaotic-systems forecasting
line (Zhang & Gilpin, [2505.11349](https://arxiv.org/abs/2505.11349);
[2409.15771](https://arxiv.org/abs/2409.15771)) finds TSFMs forecast by copying
context motifs, lose pointwise tracking within a correlation time, yet preserve
*invariant statistics*. Our mechanism analysis (Phase 7) is a sharp instance:
the tail mean **is** an invariant statistic, the rollouts copy a level and stop
tracking after ~8–22 steps (the truth's own $\tau_c$ is 8–9), and "good
few-shot" means "the copied level is right". This is why a model-free kNN-copy
baseline is already competitive, and why retrieval helps only once shared
scaling makes the level copyable.

**Collapse to the mean.** The rollouts systematically *under-disperse* (std
ratio $\le 0.6$, frequent exact flatlines) and over-smooth — the
collapse-to-the-mean behaviour analyzed for transformer ICL in
[2510.09776](https://arxiv.org/abs/2510.09776). For a level metric this collapse
is *benign* (a flat line at the right level scores perfectly); for any
dynamics-sensitive metric it would be fatal. The thesis should be explicit that
our strong few-shot numbers are a statement about level calibration, not about
turbulence dynamics.

**Mean vs median.** Because the target is a positive, right-skewed,
time-averaged flux, the RMSE-optimal point estimate is the conditional mean, not
the median (Phase 5). The per-model split (mean helps Chronos-2/Bolt/TiRex,
median wins for TimesFM) is itself diagnostic: where mean decoding helps most,
level calibration was worst — the same level story from the decoding angle.

**Limits.** Six ID / five OOD traces cap statistical power (Wilcoxon floors at
$p=0.031$ ID / $0.0625$ OOD; the bootstrap CI is primary). Headline best-config
ID numbers carry wide CIs, and the Phase-8 reconciliation (below) shows the
best-config retrieval-ICL gains are *phase-sensitive* across subsample phases
while the robust facts (finetuning's ~22 ID step, the baselines) replicate.

## Evaluation reconciliation

Our few-shot side scores the `[2::3]` subsample (266 steps); Severin's
finetuning side scores `[0::3]` (267 steps) of the *same* raw simulations, and
the chronos2 notebooks use a `mean(x[:-80])` metric that includes the copied
context. Phase 8 re-runs the ladder on the `[0::3]` phase under the honest
`[-80:]` tail so both halves sit on one internally-consistent ladder, and
confirms the saturation level is empirically phase-invariant (max per-trace
tail-mean delta < 1%). The robust ladder rungs agree across phases; the
best-config retrieval-ICL rungs are phase-sensitive (the same $n=6$ fragility
from a second angle). Full analysis:
[`docs/results/fewshot/evaluation_reconciliation.md`](../results/fewshot/evaluation_reconciliation.md)
(figure `reconciliation_ladder.png`).

## Map

| phase | doc | results |
|---|---|---|
| 2 — example selection | [example_selection.md](example_selection.md) | [selection_table.md](../results/fewshot/selection_table.md) |
| 3 — presentation format | [presentation_format.md](presentation_format.md) | [presentation_table.md](../results/fewshot/presentation_table.md) |
| 4 — OP covariates | [operating_param_covariates.md](operating_param_covariates.md) | [covariates_table.md](../results/fewshot/covariates_table.md) |
| 5 — decoding & ensembling | [decoding_and_ensembling.md](decoding_and_ensembling.md) | [decoding_table.md](../results/fewshot/decoding_table.md) |
| 6–7 — ICL × finetuning, mechanism | [icl_finetuning_synergy.md](icl_finetuning_synergy.md) | [finetuned_icl_table.md](../results/fewshot/finetuned_icl_table.md), [mechanism_table.md](../results/fewshot/mechanism_table.md) |
| 8 — evaluation reconciliation | (this doc) | [evaluation_reconciliation.md](../results/fewshot/evaluation_reconciliation.md) |
