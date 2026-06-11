# Few-Shot ICL — Improvement Roadmap

Phased plan for the few-shot in-context learning side of the project (Lukas).
Each phase is scoped to roughly one working session and is self-contained:
goal, motivation, tasks, deliverable, and pointers into the codebase.
Later phases depend on earlier ones where noted.

Context: our current pipeline (`src/fusiontimeseries/benchmarking/few_shot/`)
flat-concatenates k example context→target pairs in front of the query context,
selects examples **randomly**, and z-scores each example independently.
Best result so far: TiRex k=5 at 42.33 ID / 33.89 OOD RMSE. Supervisor input
(Fabian, Jan 9): select benchmark examples by **operating parameters** or
**nearest neighbour to context**; conditioning references FiLM
([1709.07871](https://arxiv.org/abs/1709.07871)) and DiT
([2212.09748](https://arxiv.org/abs/2212.09748)).

---

## Phase 0 — Data plumbing: operating parameters for the example pool

**Goal**: make the four operating parameters (q, ŝ, R/L_T, R/L_N) accessible
for every trace in our few-shot example pool and for the 11 test traces.

**Why**: Phases 2 and 4 retrieve/condition on operating parameters. Our pool
is built from raw `.dat` iterations (`FLUX_TRACE_DIR`), but the parameters
live in `data/flux_data.json` (each entry has `energy_flux, rlt, rln, q, shat`)
under re-coded keys (`gyroswin_train` = 1000+, `gyroswin_id` = 3000+,
`gyroswin_ood` = 4000+, plus `batch_1..10`).

**Tasks**:
- [ ] Verify the mapping between raw iteration IDs (0–299) and
      `flux_data.json` keys (match subsampled `energy_flux` values against
      our traces; the 6 ID test traces [8, 115, 131, 148, 235, 262] should
      match `gyroswin_id/3000..3005`).
- [ ] Add `operating_params` (dict or tuple) to `FewShotExample` in
      `few_shot_utils.py`; populate in `create_example_pool()`.
- [ ] Expose params for the 11 benchmark traces (extend
      `BenchmarkDataProvider` or a small lookup helper).
- [ ] Normalize params with the ranges in `lib/config.py`
      (`shat: (0,5), q: (1,9), rlt: (3.5,12), rln: (0,7)`) — reuse, don't
      re-derive.
- [ ] Smoke test: print params for one train + one ID + one OOD trace and
      sanity-check against Severin's PCA-KMeans clustering notes
      (docs/report).

**Deliverable**: example pool and test traces carry operating parameters;
a verified ID↔key mapping documented in code.

---

## Phase 1 — Evaluation harness: seeds, significance, baselines

**Goal**: a rigorous comparison harness so every later phase produces
trustworthy numbers.

**Why**: we have only 6 ID + 5 OOD test traces and stochastic example
selection. Single-seed RMSE differences of a few points are likely noise.
Also on the project-wide TODO list: baselines and statistical testing.

**Tasks**:
- [ ] Multi-seed evaluation: run each configuration over ~20 example-sampling
      seeds; report mean ± std across seeds (in addition to the per-trace SE).
- [ ] Paired significance tests between strategies: Wilcoxon signed-rank /
      paired bootstrap over per-trace errors (same traces, same seeds).
- [ ] Simple baselines, evaluated with the exact benchmark metric
      (RMSE of time-averaged tail, last 80 steps):
  - [ ] Persistence (repeat last context value)
  - [ ] Training-pool tail-mean (predict the global mean saturation level)
  - [ ] kNN-copy: retrieve nearest training trace (by context distance) and
        copy its tail — retrieval without any TSFM. This bounds what
        retrieval alone achieves and competes with the paper's GPR (43.8 ID).
- [ ] Persist results as JSON (one file per run, like the existing
      benchmark output) + a small aggregation/plot helper.

**Deliverable**: `evaluation_harness.py` (or extension of
`few_shot_utils.py`) + baseline numbers table. Every subsequent phase
reports through this harness.

---

## Phase 2 — Retrieval-based example selection  ← Fabian's explicit ask

**Goal**: replace random example selection with informed retrieval and
quantify the gain. Core scientific contribution of the few-shot side.

**Why**: random examples carry no information about the query's regime.
Similar operating parameters → similar saturation level, which is exactly
what the metric measures. Active literature to cite/frame against:
[RAF](https://arxiv.org/pdf/2411.08249), [TS-RAG](https://arxiv.org/pdf/2503.07649),
[TimeRAF](https://arxiv.org/pdf/2412.20810), [RAFT](https://arxiv.org/pdf/2511.05859).

**Tasks** (new `select_examples_*` functions beside `select_examples_random`):
- [ ] `select_examples_op_knn`: k nearest in normalized OP space
      (needs Phase 0).
- [ ] `select_examples_context_nn`: k nearest by query-context similarity —
      try (a) Euclidean on z-scored context, (b) DTW, (c) linear-phase
      growth-rate feature (physics-motivated scalar; theory links growth
      rate to saturation amplitude).
- [ ] `select_examples_oracle`: cheating selection minimizing test error —
      upper bound / headroom estimate (clearly marked as diagnostic).
- [ ] Optional: diversity-aware variant (max-marginal-relevance between
      similarity and diversity).
- [ ] Run the grid: {random, op_knn, context_nn × 3 distances, oracle} ×
      {k = 1, 3, 5, 10} × 4 models, through the Phase-1 harness.
- [ ] Analysis: does op_knn beat context_nn? (If equal → context already
      encodes the parameters; if op wins → motivates parameter conditioning.)

**Deliverable**: selection-strategy results table + significance tests;
the random-vs-retrieval-vs-oracle plot. Likely the headline figure for the
few-shot chapter.

---

## Phase 3 — Example presentation format

**Goal**: fix the known weakness of flat concatenation and re-examine the
k-curve.

**Why**: [TimesFM-ICF](https://arxiv.org/pdf/2410.24087) (ICML 2025) showed
that concatenated series without separators look like one jagged stream and
degrade ICL — they needed a learnable separator token + continued
pretraining. Our k=10 degradation is plausibly this artifact: 10 splices =
10 fake discontinuities. [Chronos-2](https://arxiv.org/abs/2510.15821) offers
the principled alternative natively.

**Tasks**:
- [ ] Chronos-2 **group ICL**: pass the k examples as related series in a
      group (its group attention shares information across series) instead
      of concatenating. Compare vs concat on identical example sets.
- [ ] Re-run the k-curve (k = 0, 1, 3, 5, 10, and beyond if it keeps
      improving) under group ICL — does k=10 degradation disappear?
- [ ] Normalization ablation: per-example z-scoring (current) vs **shared**
      scaling for examples + query. Per-example scoring destroys relative
      amplitude — the main signal examples can carry about saturation level.
- [ ] Ordering ablation (concat models): most-similar example nearest to the
      query vs furthest vs random order.
- [ ] Optional: truncated examples (only the overshoot/transition region)
      to fit more examples into fixed context budget (TiRex/TimesFM).

**Deliverable**: format comparison table; revised k-curve; the
"presentation matters" result for the thesis discussion.

---

## Phase 4 — Training-free operating-parameter conditioning (bridge phase)

**Goal**: condition forecasts on operating parameters **without any
training**, via Chronos-2's zero-shot covariate support.

**Why**: completes the adaptation ladder for the joint thesis story —
zero-shot → ICL → ICL + OP covariates → finetuned OPC (Severin) — with
monotonically increasing adaptation cost. Chronos-2 accepts past-only /
known-future / categorical covariates through group attention with no
fine-tuning ([Amazon Science](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting),
[model card](https://huggingface.co/amazon/chronos-2)).

**Tasks**:
- [ ] Pass the 4 operating parameters as constant covariate channels
      (known-future, since they're static per simulation) alongside the
      query flux series (needs Phase 0).
- [ ] Combine with ICL: examples + their covariates + query + its
      covariates in one group.
- [ ] Evaluate: zero-shot vs +covariates vs ICL vs ICL+covariates
      (k from Phase 2/3 best config), ID and OOD.
- [ ] Compare against Severin's finetuned OPC numbers (BilinearLoRA family)
      and the paper's GPR baseline — "how far does training-free
      conditioning get?"

**Deliverable**: the ladder table (adaptation cost vs RMSE) — the bridge
result connecting both sides of the project.

---

## Phase 5 — Analysis: where do the gains come from?

**Goal**: explain the mechanism behind few-shot gains, not just measure them.

**Why**: the benchmark metric (RMSE of time-averaged tail) is dominated by
the predicted saturation *level*. Understanding whether ICL calibrates
amplitude or genuinely improves dynamics shapes the thesis narrative and
justifies the retrieval design choices.

**Tasks**:
- [ ] Error decomposition per trace: bias of the predicted tail mean vs
      RMSE of the (mean-removed) fluctuations. Hypothesis: ICL gains are
      mostly amplitude calibration.
- [ ] Correlation-time / rollout-stability analysis (the GyroSwin paper
      reports ~110 steps vs ~7–10 for baselines): does ICL extend the usable
      autoregressive horizon of the TSFMs? Nobody has measured this.
- [ ] Per-trace breakdown: which of the 11 test traces benefit most, and do
      they correlate with example similarity (links back to Phase 2)?
- [ ] Forecast plots per strategy for docs/results/ (match the style of
      Severin's zeroshot/finetuning plots).

**Deliverable**: analysis notebook + figures; the "why it works" section.

---

## Phase 6 — Write-up & integration

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
      findings from NLP.

**Deliverable**: updated README + docs; thesis-ready tables and figures.

---

## Suggested session order

| Session | Phase | Depends on |
| ------- | ----- | ---------- |
| 1       | Phase 0 + Phase 1 (both are plumbing; fit in one session) | — |
| 2       | Phase 2 — retrieval | 0, 1 |
| 3       | Phase 3 — presentation format | 1 (2 helpful) |
| 4       | Phase 4 — covariate conditioning | 0, 1 |
| 5       | Phase 5 — analysis | 2–4 |
| 6       | Phase 6 — write-up | all |

## References

- TimesFM-ICF: In-Context Fine-Tuning for Time-Series Foundation Models — https://arxiv.org/pdf/2410.24087
- Chronos-2: From Univariate to Universal Forecasting — https://arxiv.org/abs/2510.15821
- RAF: Retrieval Augmented Time Series Forecasting — https://arxiv.org/pdf/2411.08249
- TS-RAG — https://arxiv.org/pdf/2503.07649
- TimeRAF — https://arxiv.org/pdf/2412.20810
- RAFT — https://arxiv.org/pdf/2511.05859
- FiLM — https://arxiv.org/abs/1709.07871
- DiT (adaLN conditioning) — https://arxiv.org/abs/2212.09748
- GyroSwin (evaluation protocol + baselines) — https://arxiv.org/abs/2510.07314
