# Few-Shot ICL — Improvement Roadmap

Phased plan for the few-shot in-context learning side of the project (Lukas).
Each phase is scoped to roughly one working session and is self-contained:
goal, motivation, tasks, deliverable, and pointers into the codebase.
Later phases depend on earlier ones where noted.

Context: the original pipeline (`src/fusiontimeseries/benchmarking/few_shot/`)
flat-concatenates k example context→target pairs in front of the query context,
selects examples **randomly**, and z-scores each example independently.
Starting point: TiRex k=5 at 42.33 ID / 33.89 OOD RMSE; best legit config
after Phases 2–3: Bolt mmr_euclid + shared scaling k=10 at 23.28 ID. Supervisor input
(Fabian, Jan 9): select benchmark examples by **operating parameters** or
**nearest neighbour to context**; conditioning references FiLM
([1709.07871](https://arxiv.org/abs/1709.07871)) and DiT
([2212.09748](https://arxiv.org/abs/2212.09748)).

---

## Phase 0 — Data plumbing: operating parameters for the example pool ✅ (2026-06-11)

**Goal**: make the four operating parameters (q, ŝ, R/L_T, R/L_N) accessible
for every trace in our few-shot example pool and for the 11 test traces.

**Why**: Phases 2 and 4 retrieve/condition on operating parameters. Our pool
is built from raw `.dat` iterations (`FLUX_TRACE_DIR`), but the parameters
live in `data/flux_data.json` (each entry has `energy_flux, rlt, rln, q, shat`)
under re-coded keys (`gyroswin_train` = 1000+, `gyroswin_id` = 3000+,
`gyroswin_ood` = 4000+, plus `batch_1..10`).

**Tasks**:
- [x] Verify the mapping between raw iteration IDs (0–300) and
      `flux_data.json` keys — done by exact value-matching in
      `few_shot/operating_params.py`; persisted as the tracked
      `operating_params_mapping.json` (iteration_8 ↔ gyroswin_id/3003 etc.;
      dump keys are PERMUTED vs raw ids, 1000+i ≠ raw i).
- [x] Add `operating_params` (dict) to `FewShotExample` in
      `few_shot_utils.py`; populated in `create_example_pool()`
      (244/245 pool examples covered — one valid raw trace has no dump
      entry; the 5 OOD dump entries have no raw counterpart. Phase-2
      op-kNN must filter `operating_params is None`).
- [x] Expose params for the 11 benchmark traces
      (`get_params_for_benchmark_trace()` in `operating_params.py`).
- [x] Normalize params with the ranges in `lib/config.py`
      (`normalize_params()` reuses `FTSConfig().op_ranges`).
- [x] Smoke test: `python -m ...few_shot.operating_params` prints params for
      one train + one ID + one OOD trace; normalized values land in [0,1].

**Deliverable**: example pool and test traces carry operating parameters;
a verified ID↔key mapping documented in code. ✅

**⚠️ Side discovery — test-set leakage (fixed)**: `create_example_pool`'s
`exclude_ids` excluded *pool positions*, but `get_valid_flux_traces()` keys
traces by an incremental counter over valid traces. All six ID test traces'
twins (raw 8/115/131/148/235/262 = pool indices 3/65/81/98/185/212) stayed in
the example pool, and position 262 was a silent no-op (pool size 246, not
245). Fixed: `exclude_ids` is now interpreted as raw ids via the mapping; the
old k-sweeps were re-run with the fixed pool (see README note). Would have
been fatal for Phase-2 kNN retrieval.

**⚠️ For Severin**: the old AutoGluon finetuning split
(`create_train_and_test_flux_ts_dataframes` in
`finetuning/preprocessing/utils.py`) excludes *nothing* — the six ID test
traces are part of its training pool too.

---

## Phase 1 — Evaluation harness: seeds, significance, baselines ✅ (2026-06-11)

**Goal**: a rigorous comparison harness so every later phase produces
trustworthy numbers.

**Why**: we have only 6 ID + 5 OOD test traces and stochastic example
selection. Single-seed RMSE differences of a few points are likely noise.
Also on the project-wide TODO list: baselines and statistical testing.

**Tasks**:
- [x] Multi-seed evaluation: `run_benchmark()` in `few_shot/harness.py`
      defaults to 20 example-sampling seeds; reports mean ± std across seeds
      (plus per-trace SE). (The fixed-pool k-sweep re-run used single seed 42
      to stay comparable to the published numbers; Phase-2 grids should use
      the 20-seed default.)
- [x] Paired significance tests: `paired_comparison()` — Wilcoxon
      signed-rank + paired bootstrap over per-trace squared errors. Note:
      at n=6/5 traces Wilcoxon's two-sided p bottoms out at 0.031/0.0625 —
      the bootstrap CI is the primary evidence, especially OOD.
- [x] Simple baselines (`few_shot/baselines.py`), exact benchmark metric,
      fixed pool:
  - [x] Persistence — 50.91 ID / 47.07 OOD
  - [x] Training-pool tail-mean — 38.65 ID / 51.54 OOD
  - [x] kNN-copy — k=1: 44.24 ID / 36.94 OOD (≈ the paper's GPR 43.8 ID);
        k=5: 34.98 ID / 39.54 OOD. Retrieval alone is a strong baseline.
- [x] Persist results as JSON: backward-compatible superset of the old
      schema (adds method/seeds/per-seed/per-trace records) in
      `results/few_shot_v2/`; `load_results()`/`results_table()` aggregate
      old and new files.

**Deliverable**: `few_shot/harness.py` + baseline numbers table. Every
subsequent phase reports through this harness. ✅

---

## Phase 2 — Retrieval-based example selection ✅ (2026-06-11) ← Fabian's explicit ask

**Goal**: replace random example selection with informed retrieval and
quantify the gain. Core scientific contribution of the few-shot side.

**Why**: random examples carry no information about the query's regime.
Similar operating parameters → similar saturation level, which is exactly
what the metric measures. Active literature to cite/frame against:
[RAF](https://arxiv.org/pdf/2411.08249), [TS-RAG](https://arxiv.org/pdf/2503.07649),
[TimeRAF](https://arxiv.org/pdf/2412.20810), [RAFT](https://arxiv.org/pdf/2511.05859).

**Tasks** (all in `few_shot/selection.py`; SelectFn registry `make_select_fn`,
most-similar example placed LAST, adjacent to the query):
- [x] `select_examples_op_knn`: k nearest in normalized OP space — 244
      candidates after filtering the one params-less pool example.
- [x] `select_examples_context_nn`: (a) Euclidean on z-scored context
      (verified identical to kNN-copy's neighbours), (b) numpy-DP DTW with
      optional Sakoe-Chiba band, (c) growth-rate feature (log-linear fit up
      to the overshoot peak; recovers synthetic γ within ~2%).
- [x] `select_examples_oracle` (`oracle_tail`): NOT clearly better than
      random on ID for 3 of 4 models (only Bolt k=1: 28.76 ID) →
      per-example z-scoring hides the level signal the oracle matches on —
      hands off to Phase 3's shared-scaling ablation.
- [x] MMR variant (`mmr_euclid`, λ=0.5): best Bolt retrieval config of the
      grid (30.23 ID at k=10).
- [x] Grid: 7 strategies × k ∈ {1,3,5,10} × 4 models (+ k=0 anchors),
      random = 20 seeds → `results/few_shot_v2_selection/` (116 files);
      runner `few_shot/run_selection_grid.py`.
- [x] Analysis (`few_shot/analyze_selection.py` → `docs/results/fewshot/`):
      **op_knn does NOT beat context_nn** (TiRex ID: ctx_euclid better by
      4.75, bootstrap CI [1.5, 10.2], Wilcoxon p=0.031) → the context
      already encodes the parameters; retrieval ≈ random on ID, small but
      consistent OOD gains (bootstrap-significant for TimesFM & Chronos-2).

**Deliverable**: selection-strategy results table + significance tests
(`docs/results/fewshot/selection_table.md`); the random-vs-retrieval-vs-
oracle plot (`docs/results/fewshot/selection_random_vs_retrieval_vs_oracle.png`). ✅

---

## Phase 3 — Example presentation format ✅ (2026-06-12)

**Goal**: fix the known weakness of flat concatenation and re-examine the
k-curve.

**Why**: [TimesFM-ICF](https://arxiv.org/pdf/2410.24087) (ICML 2025) showed
that concatenated series without separators look like one jagged stream and
degrade ICL — they needed a learnable separator token + continued
pretraining. Our k=10 degradation is plausibly this artifact: 10 splices =
10 fake discontinuities. [Chronos-2](https://arxiv.org/abs/2510.15821) offers
the principled alternative natively.

**Tasks** (all in `few_shot/presentation.py`; grid
`run_presentation_grid.py` → `results/few_shot_v3_presentation/`; analysis
`analyze_presentation.py` → `docs/results/fewshot/presentation_table.md`):
- [x] Chronos-2 **group ICL**: examples as `past_covariates` rows of one
      dict task, identical example sets vs concat (hard-asserted) — group is
      MUCH worse than concat at every k/strategy (random k=5: 79.1 vs 42.1
      ID); covariate rows barely act as demonstrations, and per-row instance
      norm means group cannot restore level either.
- [x] k-curve to k=20 under group ICL — group improves monotonically from
      the zero-shot anchor but plateaus far above concat; concat itself
      shows no k=20 degradation (chronos2 random k=20 = 38.0 ID, its best k,
      despite the 2048-step clamp).
- [x] Normalization ablation: shared (query-fit) scaling lets the example
      LEVEL reach the model — oracle_tail finally works (TimesFM 15.99 ID /
      8.10 OOD at k=10; beats random__shared with significant bootstrap CIs
      on all 4 models, both splits), random gets significantly WORSE ID for
      3/4 models (wrong levels now transfer), retrieval improves everywhere
      (best legit ID: 23.28, Bolt mmr_euclid shared k=10, vs 30.65 before).
      THE "presentation matters" result.
- [x] Ordering ablation (ctx_euclid): similar-last vs similar-first vs
      shuffled — non-factor, ±1–4 RMSE, no consistent direction.
- [x] Truncated examples (peak+64, applied post-selection): backfires under
      shared scaling (TiRex ctx_euclid k=10: 42.9 vs 30.8 full; trunc k=20
      loses to full k=10) — truncation removes the saturation tail the
      level mechanism copies from.

**Deliverable**: format comparison table; revised k-curve; the
"presentation matters" result for the thesis discussion. ✅

---

## Phase 4 — Point forecast & ensembling (cheap decoding wins) ✅ (2026-06-12, implemented as the "v5" grid)

**Goal**: re-decode the existing best configs with the RMSE-optimal point
statistic (mean, not median) and aggregate forecasts across example sets
and models.

**Why**: the metric is RMSE of a time-averaged tail of a positive, bursty,
right-skewed flux — the RMSE-optimal point forecast is the conditional
**mean**, but every model wrapper in `rerun_ksweep.py` decodes the
**median** (`Utils.median_forecast`; TiRex's `forecast()` even returns a
mean we currently discard). P50 systematically underestimates the level of
a right-skewed quantity. Ensembling: bagging TSFM forecasts reduces
variance by up to 54% ([2508.16641](https://arxiv.org/abs/2508.16641)).
Both surfaced in the 2026-06-12 deep-research pass; both are one-knob
changes through the existing harness. Caveat: shared scaling (Phase 3)
already fixed level transfer, so the mean-vs-median delta may be smaller
now than it would have been pre-Phase-3 — still nearly free to test.

**Tasks** (grid `few_shot/run_decoding_grid.py` →
`results/few_shot_v5_decoding/`; analysis `analyze_decoding.py` →
`docs/results/fewshot/decoding_table.md` + `decoding_effect.png`):
- [x] `point_stat` option (median | mean) on all wrappers in
      `rerun_ksweep.py` (`decode_point_forecast` helper). **Correction to
      the premise above**: TiRex does NOT return a mean natively — its
      second `forecast()` return is a relabeled median (`# median as
      mean` selects q0.5 by index; confirmed bit-identical at runtime by
      smoke D1). Mean = decile average (1/9)·Σ q₀.₁..q₀.₉ for all
      quantile models (biased low on right skew — it truncates beyond
      q0.1/q0.9); TimesFM additionally exposes its native mean head as
      `point_stat="meanhead"` (index 0 of its [mean, q0.1..q0.9] output,
      layout verified by smoke D2).
- [x] Re-ran anchors + best Phase-3 configs + oracle ceiling + 20-seed
      random for ALL 4 models under both decodings (36 cells, one MPS
      run; median cells bit-reproduce the v3/v4 twins). Mean improves ID
      in 14/16 cells, most where calibration is worst (Chronos-2
      zero-shot −20.4, random −5.2 sig.; new best legit 22.63 ID = Bolt
      mmr shared k=10 + mean). Per-model verdict: adopt mean for
      Chronos-2 (uniform gains) and Bolt/TiRex (small free wins); KEEP
      median for TimesFM (best config: decile mean +1.05 ID sig.,
      meanhead +2.01 ID sig. — the mean statistic itself, not the decile
      bias).
- [x] Seed-ensembling (analyzer-side, tail means are linear in the
      forecast): significantly better than per-seed scoring for every
      model and decoding (ID −1.7..−5.7, all bootstrap p ≤ 0.001), but
      its best cell (Bolt mean 34.27 ID) still loses to retrieval —
      fallback, not replacement.
- [x] Cross-model ensemble: ALL pairs + the 4-model average (supersedes
      the literal Bolt+TiRex — which is +3.0 ID WORSE than Bolt alone):
      no combination beats the best single model ID (closest
      Bolt+TimesFM −0.31 n.s.) — tail-level errors too correlated.

**Deliverable**: decoding/ensembling table; decision whether mean decoding
becomes the default for all later phases. ✅ — per-model: mean default for
Chronos-2 (incl. Phase 5's finetuned runs) + Bolt + TiRex; median for
TimesFM.

---

## Phase 5 — ICL on top of the finetuned model (synergy test) ✅ (2026-06-12, implemented as the "v6" grid, SELF-TRAINED checkpoint)

**Goal**: run our best retrieval-ICL config through Severin's finetuned
Chronos-2 (BilinearLoRA) and test whether ICL and finetuning compose.

**Why**: RAF ([2411.08249](https://arxiv.org/pdf/2411.08249)) reports
retrieval and finetuning are synergistic ("Advanced RAF" beats either
alone). Finetuned Chronos-2 BilinearLoRA sits at 13.83 ID / 4.86 OOD — if
ICL adds anything on top, that is the strongest joint result available to
the thesis; if it doesn't (or hurts: the finetuned model never saw
concatenated splices during finetuning), that is a citable negative result.
Either way it bridges both halves of the project.

**Tasks** (grid `few_shot/run_finetuned_grid.py` →
`results/few_shot_v6_finetuned/`; wrappers `few_shot/finetuned.py`;
training `finetuning/chronos2/train_bilinear.py`; analysis
`analyze_finetuned.py` → `docs/results/fewshot/finetuned_icl_table.md`):
- [x] Load the BilinearLoRA checkpoint behind the existing predict-fn
      interface — new module `few_shot/finetuned.py` (not rerun_ksweep:
      its factory contract stays frozen). No checkpoint from Severin
      (not responding) → SELF-TRAINED with his exact notebook recipe
      (`train_bilinear.py`, 4000 steps, 64 min MPS); grid is
      checkpoint-agnostic (`--checkpoint` + sha256 in every JSON), his
      weights swap in for a minutes-long re-run. Conditioning order gotcha:
      tensors use the FluxData order [shat, q, rlt, rln], gated by smoke F2b.
- [x] Sanity anchor: his exact eval protocol on our checkpoint gives
      ID 15.72 / OOD 6.03 on HIS metric (his published: 13.83 / 4.86 —
      same ballpark, different run/device/RNG). **Side discovery**: his
      notebooks score `mean(x[:-80])` — including the 80 copied
      ground-truth context steps; the honest `mean(x[-80:])` rescore of
      the SAME forecasts is ID 17.51 / **OOD 40.64** — the README chronos2
      finetuning rows (esp. OOD) are largely a metric artifact
      (`severin_anchor.json`, note-for-Severin in the table doc).
- [x] Run finetuned + best concat config (shared scaling, mmr_euclid and
      ctx_euclid, k ∈ {5, 10}) + a random__shared control — plus
      {median, mean} decoding twins, fresh base twins in-dir (bit-equal
      to v5), and ft @ the 512 training window.
- [x] `paired_comparison`: finetuned+ICL vs finetuned k=0, and vs base+ICL.

**Deliverable**: the "does adaptation stack?" table — 2×2
(base/finetuned × k=0/k-best). **Result: YES on ID, via retrieval quality.**
ft k=0 22.20 ID (mean decoding) already beats best base ICL (27.06); ft +
mmr_euclid k=5 → 18.62; clamped to the 512 training window → **15.63 ID, the
project's best legit number** (n=6 marginal gain n.s.; direction consistent).
Oracle stacks SIGNIFICANTLY (9.39 ID p=.043 / 10.89 OOD p=.002) — ICL
capacity survives finetuning, retrieval is the bottleneck; random examples
destroy the ft advantage (ft ≈ base at random k=10). OOD is finetuning's
story alone (67.94 → 34.10; legit ICL adds nothing). Full-window ICL is OOD
for the 512-trained model (win512 better in all 8 paired cells).

**Follow-ups (open, 2026-06-12)**:
- [ ] Send Severin the metric note (last section of
      `docs/results/fewshot/finetuned_icl_table.md`) — his published
      chronos2 rows are affected — and renew the `lora_weights.pt` ask;
      when it arrives, re-run is minutes via `--checkpoint` (grid is
      checkpoint-agnostic, sha256 recorded per cell).
- [ ] Robustness cell A — step-4000 vs step-200 weights: the shipped
      checkpoint is the recipe's best-eval pick (step 200 of 4000, noisy
      25-series eval). The full-model `checkpoint-4000/` still sits in the
      gitignored `outputs/chronos2-bilinear-selftrained-0/` — extract its
      `lora_state_dict` BEFORE cleaning outputs/ and run the ft headline
      cells to settle whether longer training helps or hurts the harness
      numbers.
- [ ] Robustness cell B — oracle_tail k10 + zeroshot at win512: completes
      the window block at the ceiling (k=0 is window-invariant by
      construction — only oracle is informative; cheap).

---

## Phase 6 — Training-free operating-parameter conditioning (bridge phase) ✅ (2026-06-12, implemented as the "v4" grid)

**Goal**: condition forecasts on operating parameters **without any
training**, via Chronos-2's zero-shot covariate support.

**Why**: completes the adaptation ladder for the joint thesis story —
zero-shot → ICL → ICL + OP covariates → finetuned OPC (Severin) — with
monotonically increasing adaptation cost. Chronos-2 accepts past-only /
known-future / categorical covariates through group attention with no
fine-tuning ([Amazon Science](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting),
[model card](https://huggingface.co/amazon/chronos-2)).

**Tasks** (all in `few_shot/covariates.py`; grid `run_covariates_grid.py`
→ `results/few_shot_v4_covariates/`; analysis `analyze_covariates.py` →
`docs/results/fewshot/covariates_table.md`):
- [x] Pass the 4 operating parameters as covariate channels — NOT as
      constants: Chronos-2 instance-norms each row independently
      (affine-invariant), so a constant channel's VALUE is erased exactly
      (verified bit-identical forecasts for tri-state-matched values;
      smoke S1a). Encoded instead as STEP functions over the concat ICL
      stream (example params over each segment, query params + known-future
      rows); zeroshot+constant-channels kept as the empirical degeneracy
      anchor (it improves 109.91 → 81.02 ID while provably carrying no
      parameter information — pure row-presence perturbation).
- [x] Combine with ICL: {random, op_knn, ctx_euclid, mmr_euclid,
      oracle_tail} × k{1,3,5,10} × {no-cov, +cov, permuted-control} under
      shared scaling, identical example sets hard-asserted; plus the
      group+cov block (structurally inert as predicted, slightly harmful).
- [x] Evaluate: **negative result with clean attribution** — +cov ≈
      permuted control everywhere (random k=10: 48.48 vs 47.90 ID), i.e.
      the channels add presence, not parameter information; the presence
      homogenizes every strategy toward ≈47 ID, destroying retrieval/oracle
      gains (oracle k=10: 23.31 → 47.28 ID, CI [8.1, 35.0]) while helping
      weak anchors (random OOD −6, also matched by the control).
- [x] Compare against Severin's finetuned OPC + paper GPR — adaptation
      ladder table + `docs/results/fewshot/adaptation_ladder.png`:
      training-free conditioning does NOT bridge ICL (29.40 ID best legit)
      to finetuned OPC (13.83 ID); absolute-level conditioning cannot
      survive the per-row instance norm without training.

**Deliverable**: the ladder table (adaptation cost vs RMSE) — the bridge
result connecting both sides of the project. ✅

---

## Phase 7 — Analysis: where do the gains come from?

**Goal**: explain the mechanism behind few-shot gains, not just measure them.

**Why**: the benchmark metric (RMSE of time-averaged tail) is dominated by
the predicted saturation *level*. Understanding whether ICL calibrates
amplitude or genuinely improves dynamics shapes the thesis narrative and
justifies the retrieval design choices. Frame against the context-parroting
line ([2505.11349](https://arxiv.org/abs/2505.11349),
[2409.15771](https://arxiv.org/abs/2409.15771)): TSFMs forecast chaotic
systems largely by copying context motifs, yet preserve invariant
statistics even after point forecasts fail — the tail mean is exactly such
an invariant statistic.

**Tasks**:
- [ ] Error decomposition per trace: bias of the predicted tail mean vs
      RMSE of the (mean-removed) fluctuations. Hypothesis: ICL gains are
      mostly amplitude calibration.
- [ ] Correlation-time / rollout-stability analysis (the GyroSwin paper
      reports ~110 steps vs ~7–10 for baselines): does ICL extend the usable
      autoregressive horizon of the TSFMs? Nobody has measured this.
- [ ] Per-trace breakdown: which of the 11 test traces benefit most, and do
      they correlate with example similarity (links back to Phase 2)?
- [ ] Target the ORACLE–LEGIT GAP (sharpened by Phase 5/v6: oracle 9.39 vs
      best legit 18.62 ID, 10.89 vs ~32 OOD — the pool CONTAINS the right
      examples even for OOD queries, context-distance retrieval just cannot
      find them): characterize what the oracle's picks have that ctx/mmr
      distance misses (tail level? params? phase?). Candidate selector to
      re-test ON THE FINETUNED MODEL: op_knn — uniquely motivated there
      (the model is literally conditioned on those params; Phase-2's
      "op_knn ≈ ctx" verdict was base-model only).
- [ ] Forecast plots per strategy for docs/results/ (match the style of
      Severin's zeroshot/finetuning plots).

**Deliverable**: analysis notebook + figures; the "why it works" section.

---

## Phase 8 — Write-up & integration

**Goal**: fold everything into the README, docs, and thesis material.

**Tasks**:
- [ ] Update README few-shot section: retrieval results, format results,
      ladder table; keep the GyroSwin-baseline framing.
- [ ] Method write-up in `docs/methods/` (example selection + presentation
      format), mirroring Severin's BilinearLoRA docs.
- [ ] Reconcile evaluation details with Severin's side (same metric, same
      test traces, same seeds where possible) so both halves are directly
      comparable in the thesis.
- [ ] Related-work paragraph: TimesFM-ICF, Chronos-2 ICL, retrieval-augmented
      forecasting (RAF / TS-RAG / TimeRAF / RAFT), ICL example-selection
      findings from NLP (Liu et al.
      [2101.06804](https://arxiv.org/abs/2101.06804)).
- [ ] Discussion framing from the chaotic-systems literature: context
      parroting + invariant statistics
      ([2505.11349](https://arxiv.org/abs/2505.11349),
      [2409.15771](https://arxiv.org/abs/2409.15771)), transformer-ICL
      collapse-to-the-mean theory
      ([2510.09776](https://arxiv.org/abs/2510.09776)); mean-vs-median
      point-estimate justification (RMSE → conditional mean).

**Deliverable**: updated README + docs; thesis-ready tables and figures.

---

## Suggested session order & parallelization

Phases form parallel bands separated by merge points; 7 and 8 are
sequential at the end:

```
Phase 0 (OP plumbing) ──┐           ┌── Phase 2 (retrieval) ✅ ─┐   ┌── Phase 4 (decoding/ensemble) ✅ ─┐
                        ├── merge ──┤                           ├───┼── Phase 5 (ICL × finetuned)      ─┼── Phase 7 ── Phase 8
Phase 1 (harness)     ──┘           └── Phase 3 (format) ✅    ─┘   └── Phase 6 (covariates) ✅        ─┘
        parallel ∥                                                          parallel ∥
```

| Session | Phase | Depends on | Parallel with |
| ------- | ----- | ---------- | ------------- |
| 1a      | Phase 0 — OP plumbing ✅ | — | 1b |
| 1b      | Phase 1 — harness ✅ | — | 1a |
| 2a      | Phase 2 — retrieval ✅ | 0, 1 | 2b |
| 2b      | Phase 3 — presentation format ✅ | 1 | 2a |
| 3a      | Phase 4 — decoding & ensembling ✅ | 3 | 3b, 3c |
| 3b      | Phase 5 — ICL × finetuning ✅ (self-trained checkpoint) | 2, 3 + Severin's checkpoint | 3a, 3c |
| 3c      | Phase 6 — covariate conditioning ✅ | 0, 1 (best config from 2/3) | 3a, 3b |
| 4       | Phase 7 — analysis | 2–6 | — |
| 5       | Phase 8 — write-up | all | — |

Notes for parallel worktree sessions:
- Keep each phase's code in its own module (`few_shot/harness.py`,
  `selection.py`, `presentation.py`, `covariates.py`) so merges are trivial.
- Worktrees do NOT contain untracked files: copy `.env` into the worktree;
  `FLUX_TRACE_DIR` / `BENCHMARK_SAVE_DIR` must be absolute paths. Use a
  distinct results subdirectory per phase to avoid clobbering.
- Author code in parallel, but stagger the heavy benchmark grids — they
  share one machine's GPU/memory.
- Phase 6's "ICL + covariates" combo can run with the default config
  (k=5, random) first and re-run with the best Phase 2/3 config after merge.

## References

- TimesFM-ICF: In-Context Fine-Tuning for Time-Series Foundation Models — https://arxiv.org/pdf/2410.24087
- Chronos-2: From Univariate to Universal Forecasting — https://arxiv.org/abs/2510.15821
- RAF: Retrieval Augmented Time Series Forecasting — https://arxiv.org/pdf/2411.08249
- TS-RAG — https://arxiv.org/pdf/2503.07649
- TimeRAF — https://arxiv.org/pdf/2412.20810
- RAFT — https://arxiv.org/pdf/2511.05859
- Liu et al., What Makes Good In-Context Examples for GPT-3? — https://arxiv.org/abs/2101.06804
- Modi & Pan, ensembling TSFM forecasts — https://arxiv.org/abs/2508.16641
- Context parroting (Zhang & Gilpin) — https://arxiv.org/abs/2505.11349
- Zero-shot forecasting of chaotic systems (Zhang & Gilpin) — https://arxiv.org/abs/2409.15771
- Why Do Transformers Fail to Forecast Time Series In-Context? — https://arxiv.org/abs/2510.09776
- FiLM — https://arxiv.org/abs/1709.07871
- DiT (adaLN conditioning) — https://arxiv.org/abs/2212.09748
- GyroSwin (evaluation protocol + baselines) — https://arxiv.org/abs/2510.07314
