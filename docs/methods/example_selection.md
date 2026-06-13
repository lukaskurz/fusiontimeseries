# Few-Shot Example Selection: Retrieval Strategies for In-Context Learning

How should the $k$ demonstration traces fed to a time-series foundation model
be chosen? This document describes the retrieval strategies in
[`benchmarking/few_shot/selection.py`](../../src/fusiontimeseries/benchmarking/few_shot/selection.py)
(Phase 2) and the central — initially counter-intuitive — finding: under the
naive presentation format, informed retrieval does **not** beat random
selection in-distribution.

## Motivation

The benchmark scores the RMSE of the time-averaged saturation level
$\bar{Q}$ over the last 80 timesteps. Random demonstration traces carry no
information about the query's operating regime, yet physically similar
configurations saturate at similar amplitudes — so retrieving demonstrations
that match the query *should* help the model calibrate the level it predicts.
This is also the explicit supervisor ask (Fabian, Jan 9): select examples by
**operating parameters** or by **nearest neighbour in context space**.

The shared interface is the harness `SelectFn`
$(\text{pool}, k, \text{seed}, q_{\text{ctx}}, q_{\text{key}}) \mapsto
[\text{examples}]$ (a strategy uses whichever arguments it needs). Every
strategy returns its $k$ picks with the **most similar example LAST**, adjacent
to the query in the flat ICL concatenation — recency favours the segment
nearest the query (Phase 3 ablates this and finds it a non-factor).

## Method

Let the pool be $\{e_i\}$, each example a full trace with an 80-step context
$c_i$, a target, and (where matched) operating parameters
$\theta_i = (q, \hat{s}, R/L_T, R/L_n)_i$. The query has context $c_q$ and
parameters $\theta_q$.

**`op_knn` — operating-parameter kNN.** Rank by Euclidean distance in
min-max-normalized parameter space,
$d_{\text{op}}(i) = \lVert \tilde\theta_i - \tilde\theta_q \rVert_2$, where
$\tilde\theta$ uses the `FTSConfig.op_ranges`. The one pool example without a
dump match (1 of 245) is filtered out.

**`ctx_euclid` — context Euclidean.** Rank by
$\lVert z(c_i) - z(c_q) \rVert_2$ where $z(\cdot)$ is the per-sample z-score
(the exact z-scoring `baselines.make_knn_copy_forecast` uses, so `ctx_euclid`
retrieves the same neighbours as the kNN-copy baseline — cross-checked in the
self-test).

**`ctx_dtw` — context dynamic time warping.** A dynamic-programming DTW with
absolute-difference local cost on z-scored contexts and an optional Sakoe-Chiba
band $|i-j| \le \text{band}$ (the accumulated cost is monotone, identity-zero,
symmetric, and bounded by the diagonal $L_1$ path). Robust to small temporal
phase offsets that Euclidean penalizes.

**`ctx_growth` — linear-phase growth rate.** A physics-motivated scalar
feature: the least-squares slope of $\log(\text{clip}(c, \varepsilon))$ over
$[0, \arg\max c + 1]$ (the exponential growth phase ends at the overshoot
peak), with a full-context fallback for early/noisy peaks. Rank by
$|\gamma_i - \gamma_q|$. Gyrokinetic theory links the linear growth rate to the
saturation amplitude, so $\gamma$ is a candidate regime signature.

**`mmr_euclid` — maximal marginal relevance.** Greedy selection trading query
similarity against diversity among the picks. With $\mathrm{sim}(a,b) = 1 / (1
+ \lVert z(a) - z(b) \rVert_2)$, each step picks

$$
\arg\max_{i \notin S}\;
\lambda\,\mathrm{sim}(c_q, c_i)
\;-\;(1-\lambda)\,\max_{s \in S} \mathrm{sim}(c_i, c_s),
\qquad \lambda = 0.5 .
$$

The first pick is the plain nearest neighbour; the list is reversed so it sits
last, adjacent to the query.

**`oracle_tail` — cheating diagnostic.** Ranks pool examples by
$|\,\overline{Q}_i^{\text{tail}} - \overline{Q}_q^{\text{tail}}\,|$ using the
query's **ground-truth** tail mean. This reads the test label and is never a
legitimate method — it is a headroom estimate of label-aware selection under
the current presentation format (nearest-*label*, not a model-in-the-loop upper
bound).

## Findings

Full grid (7 strategies $\times$ $k \in \{1,3,5,10\}$ $\times$ 4 models, random
over 20 seeds) and paired significance tests in
[`docs/results/fewshot/selection_table.md`](../results/fewshot/selection_table.md)
(figure: `selection_random_vs_retrieval_vs_oracle.png`).

- **Retrieval does not beat 20-seed random in-distribution.** Every
  bootstrap CI straddles zero, and — strikingly — even the cheating
  `oracle_tail` is no better than random ID for 3 of 4 models (only
  Chronos-Bolt at $k=1$ shows headroom, 28.76 ID). The diagnosis: the pipeline
  z-scores **each example independently**, so its absolute saturation level —
  the very quantity the metric and the oracle match on — never reaches the
  model. This hands off directly to [Phase 3](presentation_format.md).
- **Operating-parameter kNN does not beat context similarity** anywhere
  (Fabian's question): on TiRex ID, `ctx_euclid` is significantly better than
  `op_knn` ($\Delta = 4.75$, bootstrap CI $[1.5, 10.2]$, Wilcoxon $p=0.031$).
  The 80-step context already encodes the regime information the four
  parameters would supply.
- **Out-of-distribution, retrieval gives small but consistent gains**
  (bootstrap-significant for TimesFM and Chronos-2, e.g. `op_knn` +3.2 RMSE vs
  random for TimesFM).

The hard lesson — *which* examples you pick cannot help until *how* you present
them lets their level through — is the bridge to Phase 3.

## Relationship to prior work

Retrieval-augmented forecasting is an active line: **RAF**
([2411.08249](https://arxiv.org/pdf/2411.08249)), **TS-RAG**
([2503.07649](https://arxiv.org/pdf/2503.07649)), **TimeRAF**
([2412.20810](https://arxiv.org/pdf/2412.20810)), and **RAFT**
([2511.05859](https://arxiv.org/pdf/2511.05859)) all retrieve similar history
to condition a forecast — but typically *with* training (a retrieval-aware head
or finetuning), whereas our Phase 2 is training-free and isolates the selection
axis alone. The negative ID result echoes the NLP ICL example-selection
literature (Liu et al., *What Makes Good In-Context Examples for GPT-3?*,
[2101.06804](https://arxiv.org/abs/2101.06804)): good demonstrations matter, but
only when the model can actually use the matched feature — here the absolute
level, which the presentation format gates (Phase 3).

## Code and results

- Implementation:
  [`benchmarking/few_shot/selection.py`](../../src/fusiontimeseries/benchmarking/few_shot/selection.py)
  (`STRATEGIES`, `make_select_fn`); grid runner `run_selection_grid.py`;
  analysis `analyze_selection.py`.
- Results:
  [`docs/results/fewshot/selection_table.md`](../results/fewshot/selection_table.md)
  + `selection_random_vs_retrieval_vs_oracle.png`.
- Self-test: `python -m fusiontimeseries.benchmarking.few_shot.selection`.
- Narrative context: [`few_shot_icl.md`](few_shot_icl.md).
