# Method write-ups

Technical documentation for the two halves of the project. The few-shot ICL
docs (Lukas) are organized as a narrative overview plus one document per phase;
the LoRA-conditioning docs (Severin) describe the finetuning adapter variants.

## Few-shot in-context learning (Lukas)

Start with the narrative; the per-phase docs carry the detail, and each links to
its generated result table under [`../results/fewshot/`](../results/fewshot/).

- **[few_shot_icl.md](few_shot_icl.md)** — umbrella overview, related work,
  discussion, and the evaluation-reconciliation pointer. *The thesis draws from
  this doc.*
- [example_selection.md](example_selection.md) — Phase 2: retrieval strategies
  (op_knn / ctx_euclid / ctx_dtw / ctx_growth / mmr_euclid / oracle_tail) and
  the z-score-erases-level diagnosis.
- [presentation_format.md](presentation_format.md) — Phase 3: shared vs
  per-example scaling (the level round-trip), group ICL, ordering, truncation.
- [operating_param_covariates.md](operating_param_covariates.md) — Phase 4:
  training-free OP conditioning, instance-norm affine erasure, the clean
  negative.
- [decoding_and_ensembling.md](decoding_and_ensembling.md) — Phase 5: conditional
  mean vs median, decile-mean bias, per-model decision, seed/cross-model
  ensembling.
- [icl_finetuning_synergy.md](icl_finetuning_synergy.md) — Phases 6–7: the 2×2
  synergy, context composition (win512), and the level-calibration mechanism /
  oracle-gap information limit.

Reconciliation of the two halves' evaluation protocols (Phase 8):
[`../results/fewshot/evaluation_reconciliation.md`](../results/fewshot/evaluation_reconciliation.md).

## LoRA conditioning for finetuning (Severin)

- [BilinearLoRA.md](BilinearLoRA.md) — conditional low-rank adapter with
  bilinear (FiLM-like) interaction between input and operating parameters.
- [RSSBilinearLoRA.md](RSSBilinearLoRA.md) — rank-space scale **and** shift
  modulation.
- [OSSBilinearLoRA.md](OSSBilinearLoRA.md) — output-space shift/scale variant.
