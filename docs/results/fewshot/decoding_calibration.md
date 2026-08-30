# Decoding calibration on training traces (2026-08-18)

Validation of the 2026-06-14 quantile-decoding finding on data that is not the
benchmark. The original sweep
([`explore_decoding_sweep.py`](../../../src/fusiontimeseries/benchmarking/few_shot/explore_decoding_sweep.py))
selected the decoding quantile by argmin over the 6 ID / 5 OOD *benchmark*
traces and concluded "$q_{0.6}$ is a consistent best across all four models at
the deployable configs". [`decoding_and_ensembling.md`](../../methods/decoding_and_ensembling.md)
flagged the debt: *"still tuned on $n{=}6/5$, so it needs validation on held-out
traces before it is claimed as a method."* This is that validation.

Code: [`calibrate_decoding.py`](../../../src/fusiontimeseries/benchmarking/few_shot/calibrate_decoding.py)
(sweep) + [`analyze_decoding_calibration.py`](../../../src/fusiontimeseries/benchmarking/few_shot/analyze_decoding_calibration.py)
(analysis). Rows: `results/few_shot_v8_calibration/decoding_calibration.jsonl`.

## Protocol

**Calibration set.** The 244 non-benchmark pool traces that carry operating
parameters (`create_example_pool(exclude_ids=ID_TEST_RAW_IDS)`, minus raw 300
which has no dump entry). Each is scored under the *identical* protocol the
benchmark uses — 80-step context, autoregressive rollout to the end of the
267-step trace, metric $\bar{Q} = \text{mean}(x[-80:])$ — with
**leave-one-out** retrieval: the query's own raw id is dropped from its example
pool. Two configs, matching the 06-14 sweep: zero-shot $k{=}0$ and
`mmr_euclid` shared $k{=}5$. Single seed 42, deterministic.

**Leakage status.** The four base TSFMs never saw a GyroKinetic trace, so all
244 are held out *from the model* — a clean calibration split. The finetuned
Chronos-2 is the opposite: 241 of 244 are in `TRAIN_IDXS`, so its calibration
is **contaminated**, reported with the 3 traces outside `TRAIN_IDXS` scored
separately. The bias direction is known: the model under-predicts less on
traces it fit, so a train-calibrated quantile is a *lower bound* on the shift a
held-out query needs.

**Three selection rules**, all blind to the test labels:

| rule | criterion |
|---|---|
| `argmin` | minimize calibration RMSE (the 06-14 rule, honest split) |
| `bias0` | minimize $\lvert \text{mean}(\hat{y} - y) \rvert$ — zero the signed tail bias |
| `cov50` | drive $P(\hat{y} > y)$ to $0.5$ — 50% empirical coverage, scale-free |

**5-fold CV** stratified by tail level: each fold selects its own quantile on
4/5 of the calibration traces and is scored on the 1/5 it never saw, so the
held-out number prices in the cost of the selection itself. Note what this
does and does not fix — it prices the *selection*, not the finetuned model's
training contamination; only k-fold *retraining* would do that.

**Validity check.** The benchmark arm of this sweep reproduces the shipped
numbers exactly: Bolt zero-shot median $111.65$ / mean $109.03$ / $q_{0.80}$
$87.48$ and `mmr` $k{=}5$ median $26.35$ / mean $25.86$ / $q_{0.60}$ $21.74$
(06-14: 111.7 / 109.0 / 87.5 / 26.4 / 25.9 / 21.7); finetuned Chronos-2 @512
`mmr` $k{=}5$ mean $= 15.63$, the project's best legitimate ID cell.

## Headline: does $q_{0.6}$ survive?

At the deployable retrieval config (`mmr_euclid` shared $k{=}5$), ID tail RMSE.
The selection rule that matters is `bias0` (zero the signed tail bias) — see
[Which selection rule](#which-selection-rule-to-use) for why `argmin` on
calibration RMSE is the wrong one:

| model | `bias0` pick | ID @ `bias0` | ID mean (shipped) | $\Delta$ vs mean [95% CI] | $q^*_\text{test}$ | ID @ $q^*_\text{test}$ |
|---|---|---|---|---|---|---|
| Chronos-2 | **q0.60** | **21.28** | 27.06 | $-5.79$ $[-9.93, -1.45]$ | q0.60 | 21.28 |
| TimesFM | **q0.60** | **21.03** | 28.46 | $-7.43$ $[-15.27, +1.96]$ | q0.60 | 21.03 |
| TiRex | **q0.60** | **25.25** | 36.50 | $-11.25$ $[-19.16, -4.97]$ | q0.70 | 20.63 |
| Chronos-Bolt | mean | 25.86 | 25.86 | $0.00$ | q0.60 | 21.74 |
| ft Chronos-2 @512 | q0.50 | 18.77 | 15.63 | $+3.13$ $[-2.89, +6.46]$ | mean | 15.63 |

- **$q_{0.6}$ replicates for three of four base models.** Selected on 244
  held-out traces with no access to the test labels, `bias0` picks $q_{0.60}$
  for Chronos-2, TimesFM and TiRex, and beats the shipped decile-mean decoding
  on all three — bootstrap CI excluding zero for Chronos-2 and TiRex, not for
  TimesFM. For Chronos-2 and TimesFM the calibration pick lands on *exactly*
  the test-optimal quantile; Chronos-2's fold picks are unanimous. The 06-14
  recommendation is, for these models, now a calibrated knob rather than a
  tuned one.
- **Chronos-Bolt is the exception.** On 244 traces $q_{0.60}$ ranks third
  ($32.12$) behind median ($28.99$) and mean ($29.01$); the folds split and
  their held-out RMSEs agree to within $0.08$. Bolt's residual under-prediction
  at this config is only $-4.47$ on a $\approx 90$ mean level — there is
  essentially no bias left to correct, and its benchmark win at $q_{0.60}$ is
  an $n{=}6$ artifact. So the 06-14 phrasing "consistent best across *all four*
  models" is wrong; "three of four, and traceable to each model's residual
  bias" is right.
- **The finetuned model needs no shift at all.** Its median-decoding
  calibration bias is $-1.70$ with $P(\hat{y}>y) = 0.51$ — already level-
  calibrated, which is precisely what the finetuning bought. All three rules
  pick $q_{0.50}$, unanimously across folds. Mean still wins on the benchmark
  ($15.63$ vs $18.77$) but that delta's CI is $[-2.89, +6.46]$ — not
  significant. The contamination check behaves as predicted: on the 3 traces
  outside `TRAIN_IDXS` the argmin flips to `mean`, the direction expected if
  training traces understate the residual bias ($n{=}3$ — a hint, not evidence).
- **Two cells beat the report's best legitimate training-free number**
  ($22.63$ ID, Bolt `mmr` $k{=}10$ + mean): TimesFM $21.03$ and Chronos-2
  $21.28$, both at `mmr` $k{=}5$ with a decoding quantile chosen without test
  labels. Caveat: only the *decoding* knob is clean here — the config itself
  (`mmr_euclid` shared $k{=}5$) was still selected on the benchmark in Phase 2/3,
  so this is not an end-to-end leakage-free pipeline.

## The mechanism, now measured

06-14 asserted that "decode higher" is a global level-bias correction whose
size tracks the config's residual under-prediction. On 244 traces that is
directly measurable:

| model | config | median-decode calib bias | $q^*_\text{cal}$ |
|---|---|---|---|
| Chronos-2 | zeroshot k=0 | $-61.14$ | mean |
| Chronos-2 | mmr k=5 | $-13.61$ | q0.60 |
| Chronos-Bolt | zeroshot k=0 | $-56.92$ | q0.60 |
| Chronos-Bolt | mmr k=5 | $-4.47$ | q0.50 |
| TiRex | zeroshot k=0 | $-41.10$ | q0.70 |
| TiRex | mmr k=5 | $-6.57$ | mean |
| TimesFM | zeroshot k=0 | $-59.63$ | q0.80 |
| TimesFM | mmr k=5 | $-8.73$ | q0.60 |
| ft Chronos-2 | zeroshot k=0 | $+7.05$ | q0.50 |
| ft Chronos-2 | mmr k=5 | $-1.70$ | q0.50 |

Pearson $r = -0.822$ over 8 quantile-valued cells: the more a config
under-predicts, the higher the quantile it wants. The finetuned model, whose
bias is near zero (and *positive* at $k{=}0$), wants no shift — the clean
confirmation, because it is the one model whose level calibration was fixed by
training rather than by decoding.

## Why the benchmark disagreed: the optimum is level-conditional

The calibration set is large enough to stratify by saturation level; the
benchmark never was. Argmin quantile per calibration level quartile:

| model | config | Q1 | Q2 | Q3 | Q4 | best global | global RMSE | oracle-by-level | gap |
|---|---|---|---|---|---|---|---|---|---|
| Chronos-2 | zeroshot k=0 | q0.80 | mean | mean | q0.70 | mean | 58.18 | 57.61 | $+0.57$ |
| Chronos-2 | mmr k=5 | q0.50 | q0.50 | q0.60 | q0.70 | q0.60 | 25.93 | 21.10 | $+4.84$ |
| Chronos-Bolt | zeroshot k=0 | q0.70 | q0.60 | q0.60 | q0.70 | q0.60 | 61.72 | 60.78 | $+0.93$ |
| Chronos-Bolt | mmr k=5 | q0.50 | q0.50 | mean | q0.60 | q0.50 | 28.99 | 25.94 | $+3.05$ |
| TiRex | zeroshot k=0 | q0.70 | q0.70 | q0.70 | q0.70 | q0.70 | 31.26 | 31.26 | $-0.00$ |
| TiRex | mmr k=5 | mean | q0.50 | mean | q0.70 | mean | 25.91 | 23.18 | $+2.73$ |
| TimesFM | zeroshot k=0 | q0.80 | q0.80 | q0.80 | q0.80 | q0.80 | 38.13 | 38.13 | $+0.00$ |
| TimesFM | mmr k=5 | q0.50 | meanhead | q0.60 | q0.70 | q0.60 | 29.13 | 23.20 | $+5.93$ |
| ft Chronos-2 | zeroshot k=0 | q0.50 | q0.50 | q0.50 | q0.50 | q0.50 | 22.03 | 22.03 | $+0.00$ |
| ft Chronos-2 | mmr k=5 | q0.50 | q0.50 | q0.50 | q0.70 | q0.50 | 27.50 | 25.56 | $+1.94$ |

The optimal quantile **rises with the query's saturation level** — for
Chronos-2 at `mmr` $k{=}5$: q0.50 → q0.50 → q0.60 → q0.70 across quartiles.
And the benchmark's ID traces are level-skewed: at pool percentiles
$29\%, 31\%, 64\%, 85\%, 87\%, 87\%$, four of six sit in the upper third —
precisely the regime that wants the shift (OOD is worse: $26\%$ to $98\%$).
So part of what the 06-14 sweep measured was its own test set's level
composition.

This is the project's central bottleneck in a new place. A level-aware decoder
would gain $2$–$6$ RMSE over the best global quantile at exactly the deployable
configs (and $\approx 0$ at zero-shot, where the global bias dominates) — but
setting it per query requires knowing the query's level, the same quantity
[`level_matching.md`](level_matching.md) shows no distance computable from the
80-step context can recover. `oracle-by-level` is therefore a ceiling of the
same family as `oracle_tail`, not a deployable rule.

## Which selection rule to use

`argmin` on calibration RMSE is **not** the right rule — it is scale-dominated
by the high-level traces, which carry the largest squared errors. Zeroing the
signed bias transfers better, and TiRex is the decisive case:

| model | config | argmin → ID | `bias0` → ID | test-argmin → ID |
|---|---|---|---|---|
| TiRex | mmr k=5 | mean → **36.50** | q0.60 → **25.25** | q0.70 → 20.63 |
| Chronos-2 | zeroshot k=0 | mean → 89.51 | q0.80 → **67.23** | q0.80 → 67.23 |
| Chronos-Bolt | zeroshot k=0 | q0.60 → 103.88 | q0.80 → **87.48** | q0.80 → 87.48 |
| TimesFM | zeroshot k=0 | q0.80 → 66.86 | q0.80 → 66.86 | q0.90 → 44.44 |

At TiRex's retrieval config `argmin` picks `mean` and lands on $36.50$ ID while
`bias0` picks $q_{0.60}$ and lands on $25.25$ — an $11.25$ RMSE swing decided
entirely by the selection criterion, on identical forecasts. At both Chronos
zero-shot anchors `bias0` recovers the *test-optimal* quantile without seeing
the test labels. `cov50` (scale-free 50% coverage) tracks `bias0` closely but
is noisier on the small grids — it costs TiRex $10$ RMSE at `mmr` $k{=}5$.

**Head-to-head over all 10 (model, config) cells.** The two rules pick the
*same* spec in 6 of 10; where they differ, `bias0` wins ID 4/4 — but loses OOD
in 3 of those 4:

| | `bias0` better | tie | `argmin` better |
|---|---|---|---|
| ID | 4 | 6 | 0 |
| OOD | 1 | 6 | 3 |

**Recommendation: select the decoding quantile by zeroing the mean signed tail
bias on calibration traces** — a one-parameter estimator with a stated target
rather than a grid search against a held-out score. But it is explicitly an
*ID-side* calibration: it never loses on ID and sometimes wins large, while on
OOD it is neutral-to-slightly-harmful (Chronos-2 zero-shot $80.13$ vs mean
$67.94$). The two splits want different corrections — the ID/OOD retriever
split of [`selection_table.md`](selection_table.md) reappearing in the decoding
knob. Do not claim `bias0` as a uniform improvement.

## What this does and does not license

- **Claimable**: for Chronos-2, TimesFM and TiRex, decoding at $q_{0.6}$
  instead of the decile-mean at the retrieval configs is a calibrated choice
  validated on 244 held-out traces, worth $5.8$–$11.3$ ID RMSE (CI excluding
  zero for Chronos-2 and TiRex). Select it by zeroing the calibration bias, not
  by argmin.
- **Retracted**: "$q_{0.6}$ is a consistent best across *all four* models."
  Three of four, not four; Bolt's is an $n{=}6$ artifact and the finetuned
  model needs no shift.
- **New**: the optimum is level-conditional, and the benchmark ID set is
  level-skewed — so a slice of the original effect was test-set composition.
- **Still open**: the finetuned calibration is contaminated (241/244 traces are
  its training data). Removing that needs $K$ LoRA retrains with per-fold
  held-out calibration ($\approx 1$h/checkpoint on an M1 Max). Worth doing only
  if the ft decoding choice is to be claimed as a method; the current evidence
  (bias $\approx 0$, mean-vs-median delta not significant) says the choice
  barely matters for the ft model.
- **Nothing published is wrong**: the report ships mean decoding throughout
  (`04-results.tex`), never $q_{0.6}$, so no headline number needs revising.

## Reproduce

```bash
uv run python -m fusiontimeseries.benchmarking.few_shot.calibrate_decoding \
    --device mps --models chronos_bolt ft_chronos2 chronos2 timesfm tirex
uv run python -m fusiontimeseries.benchmarking.few_shot.analyze_decoding_calibration \
    --out docs/results/fewshot/decoding_calibration_report.md
```

The sweep is resumable — completed `(model, evalset, config, spec)` groups are
skipped on re-invocation.
