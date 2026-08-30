# Progress note — few-shot ICL improvement program (2026-06-11 → 2026-06-13)

*Source material for a progress email. Heat-flux time-series prediction with
pre-trained foundation models (TiRex / Chronos-2 / Chronos-Bolt / TimesFM) on
GKW gyrokinetic turbulence; zero-shot, few-shot in-context learning (ICL), and
LoRA finetuning, evaluated in-distribution (ID) vs out-of-distribution (OOD).*

## TL;DR — the arc of the last few days

Over the last three days I took the few-shot ICL side of the project from "the
benchmark numbers can't be trusted yet" to a complete, mechanistically-explained
story with a concrete recommendation for what to do next. The work runs as a
sequence of experiments, each answering one question:

- **Foundation** — fixed a test-set leakage bug, built a proper evaluation
  harness (multi-seed, paired bootstrap significance) and model-free baselines.
- **Can informed example *selection* beat random?** Not by itself — and the
  reason was diagnostic.
- **Fixing the *presentation* (shared scaling)** unlocked the level signal and
  turned retrieval into a real win.
- **Training-free operating-parameter conditioning (covariates)** — a clean
  negative, with a proof of *why*.
- **Decoding & ensembling** — mean decoding is a free, principled gain.
- **Finetuning × ICL** — adaptation stacks, but only through retrieval quality.
- **Mechanism** — every gain on this benchmark is *level calibration*; the
  ceiling is an information limit of the short context.
- **Reconciliation & a metric audit** — put both halves of the project on one
  consistent ladder and caught a scoring discrepancy in the finetuning numbers.
- **Level-aware retrieval** — matching on *level* (not shape) is the right
  signal, especially OOD, and it works on the finetuned model too.
- **In-context finetuning (ICF)** — training the adapter *on demonstrations*
  proves the model can learn to use level-matched demos, but does not beat the
  inherited ability under realistic retrieval.

**The single through-line:** *the model's in-context capacity was never the
bottleneck — retrieval is.* Foundation models forecast this chaotic system by
copying context motifs and preserve the saturation level (the invariant our
metric scores); the whole game is getting a **level-matched** example in front
of the model, and an 80-step context can't reliably identify one. The next
lever is therefore better **level estimation / side-information retrieval**, not
a bigger model or more in-context training.

**Standing caveat:** the benchmark has only **6 ID and 5 OOD traces**, so most
single comparisons are not individually significant. I report bootstrap
confidence intervals throughout and flag the few results that *are* robust.

---

## 0–1. Foundation: trustworthy evaluation first

- **Test-set leakage fix.** The example pool was excluding test traces by
  *position* instead of by raw simulation id, which leaked the six ID test
  traces' own twins back into the candidate pool. Fixed (pool 246 → 245,
  exclusion by raw id). Effect: k≥3 numbers got *worse* after the fix (e.g.
  TiRex k=5 ID 30.5 → 38.5) — i.e. some of the prior "good" few-shot numbers
  were partly leakage. Everything below uses the clean pool.
- **Evaluation harness + baselines.** A multi-seed runner with per-trace
  records, paired bootstrap + Wilcoxon significance tests, and three model-free
  baselines (persistence; the pool-mean constant; k-nearest-neighbour copy).
  These anchor every later comparison.

---

## 2–9. The experiments, as a sequence of questions

### Phase 2 — Does informed example *selection* beat random?
- Tried operating-parameter k-NN, several context-similarity distances, a
  diversity-aware variant, and a label-aware "oracle" (a cheating ceiling).
- **Result: no — retrieval ≈ random on ID, and even the cheating oracle didn't
  help for 3 of 4 models.** Diagnosis (the important part): the ICL pipeline
  z-scored every example independently, which *erases the absolute level* — the
  very thing the metric scores. The selection wasn't the problem; the
  presentation was.

### Phase 3 — Fix the presentation: shared scaling (**the unlock**)
- Normalize the examples and the query with **one** scaler fit on the query, so
  an example's absolute level survives into the model.
- **Result: the picture inverts.** The oracle now works and beats shared-norm
  random with bootstrap-significant margins on **all four models, both splits**
  (e.g. TimesFM 16.0 ID / 8.1 OOD at k=10). Best legitimate training-free ID
  improved 30.7 → 23.3 (Bolt). Random selection under shared scaling gets
  *worse* (wrong levels now transfer too), so shared scaling only makes sense
  *with* retrieval. (Also: Chronos-2 "group" ICL is much worse than flat concat;
  example ordering is a non-factor; truncation backfires.)

### Phase 4 — Training-free operating-parameter conditioning (covariates)
- Feed the operating parameters (q, shat, R/L_T, R/L_n) as extra covariate
  channels to Chronos-2, no training.
- **Result: a clean negative — and we can prove why.** Chronos-2 instance-norms
  each channel independently, so constant parameter channels are *value-erased*
  exactly; step-encoded channels behave like a *permuted-parameter control*
  (the model reacts to their *presence*, not their information). It homogenizes
  every strategy toward ~47 ID and destroys the oracle. Useful as a bridge rung
  on the adaptation ladder, but conditioning has to happen *in training*, not at
  inference.

### Phase 5 — Decoding & ensembling
- **Mean decoding** (averaging the quantiles) instead of the median improves ID
  in 14 of 16 cells, most where calibration is worst (Chronos-2 zero-shot
  −20). New best legitimate training-free cell: **22.6 ID**. Adopted a per-model
  default (mean for Chronos-2/Bolt/TiRex, median for TimesFM).
- Seed-ensembling helps but loses to retrieval; cross-model ensembling never
  beats the best single model.

### Phase 6 — Does adaptation stack? (finetuning × ICL)
- Self-trained a LoRA adapter with operating-parameter conditioning (the
  group's exact recipe), then ran the retrieval-ICL configurations on top.
- **Result: adaptation stacks on ID, *through retrieval quality*.** Finetuning
  alone (no demos) already beats the best base ICL; adding retrieval-ICL takes
  it to **15.6 ID at the training window — the project's best legitimate
  number**. The cheating oracle stacks *significantly* on the finetuned model
  (9.4 ID / 10.9 OOD), and random demos *erase* the finetuned advantage — so the
  in-context capacity survived finetuning and **the bottleneck is retrieval
  quality, not the model**. OOD is finetuning's story (68 → 34). (Robustness:
  the early best-eval checkpoint beats the fully-trained one everywhere —
  overtraining degrades in-context ability the most.)

### Phase 7 — Why do the gains happen? (mechanism)
- Re-ran the headline cells with full forecast trajectories dumped and
  decomposed the error exactly into a level-bias term and a fluctuation term.
- **Result: every gain is level calibration.** The level term absorbs ~90–110%
  of every configuration's error change; the fluctuation error sits at the
  chaos floor for *every* configuration (nothing actually phase-tracks the
  turbulence); and the benchmark RMSE *is* the level bias. **No genuine horizon
  extension** — forecasts track for ~8–9 steps (the truth's own correlation
  time), then collapse to a level, under-dispersed and over-smoothed. The
  **oracle gap is an information limit of the 80-step context**: the pool
  contains a near-exact level twin for almost every query, but every retrieval
  distance z-scores the context and erases the level signal, so the oracle's
  picks rank ~98/245 under context distance. Closing the gap needs side
  information, not a better distance over the same 80 steps.

### Phase 8 — One ladder across both halves, and a metric audit
- The few-shot side and the finetuning side score the *same* raw simulations at
  different subsample phases; I re-ran the adaptation ladder on the finetuning
  side's phase so both halves sit on one consistent ladder.
- **Result:** the saturation level is empirically *phase-invariant* (< 1%
  difference), so the two halves are directly comparable; the load-bearing rungs
  (finetuning, baselines) replicate, while the best-config retrieval-ICL rung is
  phase-sensitive (reinforcing the n=6 non-significance of that marginal gain).
- **Metric audit (worth flagging):** the Chronos-2 finetuning notebooks score
  the mean over everything *except* the last 80 steps — which *includes* the 80
  ground-truth context steps the forecast copies verbatim — so the very low
  published finetuning OOD numbers are largely a scoring artifact. The honest
  tail rescore of the *same* forecasts gives ~17.5 ID / 40.6 OOD. (Documented
  with a note for the maintainer; the TimesFM rows are unaffected.)

### Level-matching follow-up — *level*, not *shape*
- The k-NN baseline's distance matched example *shape* and discarded level.
  Matching on **level alone** (`|mean(ctx) − mean(query)|`) is best on both
  splits and far better OOD (k-NN-copy 39.5 → 28.5 OOD); the new `ctx_level`
  retriever reaches the project's lowest OOD in the Bolt pipeline. Direct
  model-free confirmation of the Phase-7 finding that the early-context mean is
  the strongest level predictor.

### Part A — Level-aware retrieval on the *finetuned* model
- The earlier finetuning study only used a shape retriever. I added `ctx_level`
  and a level+diversity hybrid and re-ran.
- **Result: the level-vs-shape split holds on the finetuned model.**
  `ctx_level` **significantly improves the finetuned model's OOD** (34.1 → 27.4,
  bootstrap p≈0.000; → **24.6 at the training window** = its best legitimate
  OOD), overturning the earlier "nothing improves OOD" reading; shape matching
  still wins ID (18.6). **Shape for ID, level for OOD — no single 80-step
  retriever wins both.**

### Phase 9 — In-context finetuning (ICF)
- Trained the adapter **on multi-example demonstration concatenations** (not
  single traces), with a **level-matched checkpoint** and a **random-demo
  control** — the control tells us whether the model learned to use
  level-matched demos specifically.
- **Result (clean and conclusive on one axis):**
  - **ICF makes the model demonstration-dependent** — it collapses without demos
    and gains hugely with them (≈ −29 ID, p≈0.01). So it clearly learned to
    *use* demonstrations.
  - **The control proves the usage is level-specific (the headline).** Given
    perfectly level-matched demos, the level model beats the random control on
    OOD by ~8× — **7.5 vs 61.8 RMSE, bootstrap p≈0.000** — and the random model
    gains nothing from level-matched demos (it learned to ignore level). It even
    beats the single-trace model's oracle ceiling on OOD → ICF moved the
    *ceiling*.
  - **But it does not beat the inherited ability under realistic retrieval** —
    statistically tied with the single-trace finetuned model; the gain appears
    only with oracle (label-aware) demos; the project's best legitimate number
    is unbeaten.
  - **A telling ID/OOD personality split**: the random control gets the best ID
    number we've seen (16.4) but catastrophic OOD (57) — a level-blind amplitude
    trick; the level model is balanced.
- **Conclusion:** the bottleneck was never in-context capacity (pretrained *or*
  ICF-trained) — it's retrieval.

---

## The single through-line (for the email's main point)

The mechanism is consistent across every experiment: **gains = level
calibration; the binding constraint = retrieval, not the model.** Every lever
that improved things did so by getting the right *level* in front of the model
(shared scaling, level retrieval, finetuning's learned level prior); every lever
that targeted the model's capacity instead (covariates at inference, in-context
training) either failed or hit the same retrieval wall. The concrete next step
that drops out: **a trained level predictor / side-information-driven retrieval**
to supply the level-matched demonstration the model already knows how to use.

---

## Honest caveats (state these)

- **n = 6 ID / 5 OOD traces** — most head-to-head differences are not
  individually significant; bootstrap CIs reported. The robust, strongly
  significant results are the shared-scaling oracle effect (Phase 3) and the
  OOD oracle separation in ICF (Phase 9, p≈0.000).
- Self-trained, recipe-faithful adapter (not the maintainer's exact weights —
  but they swap in for a re-run; sha recorded in every result).
- Single training run per finetuned/ICF checkpoint (best-eval selection).
- The finetuning-notebook absolute numbers use a different (easier) scoring
  window — see the metric audit; our tables use the honest tail.

---

## Where everything lives

- **Results:** `results/few_shot_v2_*` (baselines, selection), `…v3` (presentation),
  `…v4` (covariates), `…v5` (decoding), `…v6` (finetuning×ICL + Part A),
  `…v7` (mechanism), `…v8` (reconciliation), `…v9` (ICF).
- **Write-ups:** `docs/results/fewshot/` (per-phase tables + figures, incl.
  `finetuned_icl_table.md`, `mechanism_table.md`, `evaluation_reconciliation.md`,
  `level_matching.md`, `icl_finetuning_table.md`) and `docs/methods/` (narrative
  + method docs, index in `docs/methods/README.md`).
- **README** has the current headline tables and the adaptation ladder.
- All committed to `main` (≈ 50 commits, 2026-06-11 → 06-13).

---

## Draft email skeleton (edit freely)

> **Subject:** Progress — few-shot ICL: mechanism, finetuning, and in-context finetuning
>
> Hi Fabian,
>
> Big push on the few-shot side over the last few days — summary of where it
> landed.
>
> First I fixed a test-set leakage bug in the example pool and built a proper
> evaluation harness (multi-seed + bootstrap significance) and baselines, so the
> numbers are now trustworthy (some prior "good" few-shot results were partly
> leakage).
>
> The core finding: **every gain on this benchmark is level calibration**
> (matching the saturation amplitude), and **the bottleneck is retrieval, not the
> model.** Concretely:
> - Informed example *selection* alone didn't beat random — because the pipeline
>   was z-scoring away the level. Fixing the presentation (**shared scaling**)
>   turned retrieval into a real win and made the label-aware oracle work on all
>   four models.
> - **Matching examples by level** (not shape) is the right signal, especially
>   OOD; it even improves the finetuned model's OOD significantly (34 → 25).
> - **Finetuning × ICL stacks** (best legitimate ID ≈ 15.6), but the oracle and
>   the random-demo controls show the limiting factor is retrieval quality, not
>   the model's in-context ability.
> - A mechanism analysis confirms there's **no genuine horizon extension** — the
>   models track for ~8–9 steps then collapse to a level; the metric is an
>   invariant statistic they preserve.
> - I also trained the adapter **on demonstrations** (in-context finetuning):
>   with a level-vs-random control it's clear the model *learns* to use
>   level-matched demos (OOD 7.5 vs 61.8 with ideal demos, p≈0.000), but it
>   doesn't beat the inherited ability under realistic retrieval — same wall.
> - Side note: the published Chronos-2 finetuning OOD numbers are largely a
>   scoring-window artifact (they include the copied context); honest rescore is
>   ~17.5 ID / 40.6 OOD.
>
> So the recommendation is to invest next in **better level estimation /
> side-information retrieval** rather than more in-context training or a bigger
> model. Caveat throughout: only 6 ID / 5 OOD traces, so I report CIs; the two
> oracle results above are the robust ones. Everything's coded, run, documented,
> and committed — happy to walk through the tables/figures.
>
> Best,
> Lukas
