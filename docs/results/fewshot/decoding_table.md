# Point-forecast decoding and ensembling — Phase 5 results

All four models, t266 protocol: fixed 245-trace pool, full-length example targets, context 80, prediction 64, tail 80, flat concat + SHARED scaling (the Phase-3 winner) everywhere. `median` is the frozen Phase-1..4 decoding (q0.5 of the 9 deciles); `mean` is the decile-average estimator (1/9)·Σ q₀.₁..q₀.₉ — biased LOW on right-skewed data because it truncates the tails beyond q0.1/q0.9; `meanhead` is TimesFM's native mean output head (index 0 of its [mean, q0.1..q0.9] forecast — the unbiased-head cross-check; the other models have no native mean: TiRex's library "mean" return is a relabeled median, confirmed at runtime by smoke D1). Cells are `ID / OOD` tail RMSE; the 20-seed random cells show mean±std over seeds. All cells from ONE grid run (`results/few_shot_v5_decoding/`); identical example sets across decoding twins (hard-asserted).

## Decoding effect per model and config


### NX-AI/TiRex

| config | median | mean | Δmean−median (ID) | Δ (OOD) |
|---|---|---|---|---|
| zero-shot k=0 | 78.92 / 62.49 | 75.67 / 59.99 | -3.25 | -2.50 |
| best: ctx_euclid k=10 | 30.77 / 38.62 | 30.68 / 37.95 | -0.09 | -0.68 |
| oracle_tail k=10 (cheats) | 30.65 / 18.01 | 31.00 / 18.86 | +0.35 | +0.85 |
| random k=10 (20 seeds) | 44.61±13.65 / 48.56±8.15 | 43.91±13.44 / 48.49±8.12 | -0.70 | -0.07 |

### google/timesfm-2.5-200m-pytorch

| config | median | mean | Δmean−median (ID) | Δ (OOD) | meanhead | Δmeanhead−median (ID) | Δ (OOD) |
|---|---|---|---|---|---|---|---|
| zero-shot k=0 | 97.54 / 87.70 | 95.88 / 83.30 | -1.66 | -4.40 | 96.22 / 83.19 | -1.32 | -4.51 |
| best: mmr_euclid k=10 | 23.85 / 37.62 | 24.91 / 39.19 | +1.05 | +1.57 | 25.86 / 41.08 | +2.01 | +3.46 |
| oracle_tail k=10 (cheats) | 15.99 / 8.10 | 15.83 / 9.69 | -0.16 | +1.58 | 16.73 / 11.04 | +0.74 | +2.94 |
| random k=10 (20 seeds) | 45.73±11.35 / 53.24±8.31 | 45.36±11.44 / 53.95±8.33 | -0.37 | +0.71 | 44.80±11.72 / 53.86±8.46 | -0.93 | +0.62 |

### amazon/chronos-2

| config | median | mean | Δmean−median (ID) | Δ (OOD) |
|---|---|---|---|---|
| zero-shot k=0 | 109.91 / 85.86 | 89.51 / 67.94 | -20.40 | -17.92 |
| best: mmr_euclid k=5 | 29.40 / 42.56 | 27.06 / 42.64 | -2.34 | +0.08 |
| oracle_tail k=10 (cheats) | 23.31 / 23.99 | 21.61 / 21.83 | -1.70 | -2.15 |
| random k=10 (20 seeds) | 44.42±8.36 / 49.18±6.63 | 39.18±8.23 / 46.07±6.35 | -5.24 | -3.11 |

### amazon/chronos-bolt-tiny

| config | median | mean | Δmean−median (ID) | Δ (OOD) |
|---|---|---|---|---|
| zero-shot k=0 | 111.65 / 90.69 | 109.03 / 92.32 | -2.62 | +1.63 |
| best: mmr_euclid k=10 | 23.28 / 37.81 | 22.63 / 38.68 | -0.65 | +0.87 |
| oracle_tail k=10 (cheats) | 26.01 / 9.81 | 24.73 / 10.02 | -1.28 | +0.22 |
| random k=10 (20 seeds) | 38.84±12.97 / 45.16±5.41 | 37.89±12.85 / 45.78±5.27 | -0.95 | +0.62 |

![decoding_effect](decoding_effect.png)

## Paired comparisons — mean vs median per (model, config)

Δ = RMSE(A) − RMSE(B); negative favours A (mean better). CI/p from a 10k paired bootstrap over traces; Wilcoxon shown for completeness (p floors 0.031 ID / 0.0625 OOD). `[trace_seed]` rows appear for the multi-seed random cells.

| A vs B | split | RMSE A | RMSE B | Δ(A−B) | 95% CI | p_boot | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| tirex zero-shot k=0: mean vs median | ID | 75.67 | 78.92 | -3.25 | [-4.89, -2.52] | 0.000 | 0.031 |
| tirex zero-shot k=0: mean vs median | OOD | 59.99 | 62.49 | -2.50 | [-7.03, 0.32] | 0.085 | 0.312 |
| tirex best: ctx_euclid k=10: mean vs median | ID | 30.68 | 30.77 | -0.09 | [-0.86, 0.81] | 0.921 | 1.000 |
| tirex best: ctx_euclid k=10: mean vs median | OOD | 37.95 | 38.62 | -0.68 | [-0.93, -0.21] | 0.014 | 0.125 |
| tirex oracle_tail k=10 (cheats): mean vs median | ID | 31.00 | 30.65 | +0.35 | [-0.34, 0.73] | 0.219 | 0.438 |
| tirex oracle_tail k=10 (cheats): mean vs median | OOD | 18.86 | 18.01 | +0.85 | [-0.36, 2.36] | 0.617 | 1.000 |
| tirex random k=10 (20 seeds): mean vs median | ID | 45.82 | 46.55 | -0.73 | [-1.10, -0.47] | 0.000 | 0.031 |
| tirex random k=10 (20 seeds): mean vs median [trace_seed] | ID | 45.82 | 46.55 | -0.73 | [-1.09, -0.42] | 0.000 | 0.000 |
| tirex random k=10 (20 seeds): mean vs median | OOD | 49.13 | 49.20 | -0.07 | [-0.46, 0.36] | 0.766 | 1.000 |
| tirex random k=10 (20 seeds): mean vs median [trace_seed] | OOD | 49.13 | 49.20 | -0.07 | [-0.37, 0.24] | 0.619 | 0.419 |
| timesfm zero-shot k=0: mean vs median | ID | 95.88 | 97.54 | -1.66 | [-8.19, 5.39] | 0.668 | 0.438 |
| timesfm zero-shot k=0: mean vs median | OOD | 83.30 | 87.70 | -4.40 | [-9.42, 4.09] | 0.209 | 0.438 |
| timesfm best: mmr_euclid k=10: mean vs median | ID | 24.91 | 23.85 | +1.05 | [0.29, 2.13] | 0.018 | 0.156 |
| timesfm best: mmr_euclid k=10: mean vs median | OOD | 39.19 | 37.62 | +1.57 | [0.10, 3.42] | 0.019 | 0.312 |
| timesfm oracle_tail k=10 (cheats): mean vs median | ID | 15.83 | 15.99 | -0.16 | [-0.90, 0.85] | 0.881 | 0.688 |
| timesfm oracle_tail k=10 (cheats): mean vs median | OOD | 9.69 | 8.10 | +1.58 | [-0.07, 3.21] | 0.064 | 0.188 |
| timesfm random k=10 (20 seeds): mean vs median | ID | 46.71 | 47.05 | -0.34 | [-0.75, 0.06] | 0.121 | 0.312 |
| timesfm random k=10 (20 seeds): mean vs median [trace_seed] | ID | 46.71 | 47.05 | -0.34 | [-0.80, 0.15] | 0.172 | 0.063 |
| timesfm random k=10 (20 seeds): mean vs median | OOD | 54.55 | 53.85 | +0.71 | [-0.88, 2.24] | 0.520 | 0.438 |
| timesfm random k=10 (20 seeds): mean vs median [trace_seed] | OOD | 54.55 | 53.85 | +0.71 | [-0.05, 1.53] | 0.069 | 0.086 |
| chronos2 zero-shot k=0: mean vs median | ID | 89.51 | 109.91 | -20.40 | [-30.22, -12.54] | 0.000 | 0.031 |
| chronos2 zero-shot k=0: mean vs median | OOD | 67.94 | 85.86 | -17.92 | [-23.99, -12.96] | 0.000 | 0.062 |
| chronos2 best: mmr_euclid k=5: mean vs median | ID | 27.06 | 29.40 | -2.34 | [-7.39, 2.53] | 0.275 | 0.438 |
| chronos2 best: mmr_euclid k=5: mean vs median | OOD | 42.64 | 42.56 | +0.08 | [-3.00, 4.00] | 0.871 | 0.625 |
| chronos2 oracle_tail k=10 (cheats): mean vs median | ID | 21.61 | 23.31 | -1.70 | [-3.81, -0.21] | 0.025 | 0.156 |
| chronos2 oracle_tail k=10 (cheats): mean vs median | OOD | 21.83 | 23.99 | -2.15 | [-8.72, 0.38] | 0.130 | 0.625 |
| chronos2 random k=10 (20 seeds): mean vs median | ID | 39.99 | 45.16 | -5.17 | [-7.31, -2.15] | 0.001 | 0.062 |
| chronos2 random k=10 (20 seeds): mean vs median [trace_seed] | ID | 39.99 | 45.16 | -5.17 | [-6.45, -3.85] | 0.000 | 0.000 |
| chronos2 random k=10 (20 seeds): mean vs median | OOD | 46.48 | 49.60 | -3.12 | [-6.83, 6.53] | 0.330 | 0.812 |
| chronos2 random k=10 (20 seeds): mean vs median [trace_seed] | OOD | 46.48 | 49.60 | -3.12 | [-4.75, -1.41] | 0.000 | 0.078 |
| chronos_bolt zero-shot k=0: mean vs median | ID | 109.03 | 111.65 | -2.62 | [-4.09, -1.12] | 0.000 | 0.031 |
| chronos_bolt zero-shot k=0: mean vs median | OOD | 92.32 | 90.69 | +1.63 | [-1.63, 7.82] | 0.660 | 0.625 |
| chronos_bolt best: mmr_euclid k=10: mean vs median | ID | 22.63 | 23.28 | -0.65 | [-2.44, 0.74] | 0.385 | 0.562 |
| chronos_bolt best: mmr_euclid k=10: mean vs median | OOD | 38.68 | 37.81 | +0.87 | [-0.03, 2.57] | 0.065 | 0.188 |
| chronos_bolt oracle_tail k=10 (cheats): mean vs median | ID | 24.73 | 26.01 | -1.28 | [-2.43, 1.54] | 0.219 | 0.688 |
| chronos_bolt oracle_tail k=10 (cheats): mean vs median | OOD | 10.02 | 9.81 | +0.22 | [-0.73, 0.97] | 0.629 | 0.812 |
| chronos_bolt random k=10 (20 seeds): mean vs median | ID | 39.91 | 40.84 | -0.93 | [-1.70, 0.21] | 0.113 | 0.219 |
| chronos_bolt random k=10 (20 seeds): mean vs median [trace_seed] | ID | 39.91 | 40.84 | -0.93 | [-1.34, -0.53] | 0.000 | 0.000 |
| chronos_bolt random k=10 (20 seeds): mean vs median | OOD | 46.07 | 45.47 | +0.60 | [-1.20, 2.49] | 0.495 | 0.625 |
| chronos_bolt random k=10 (20 seeds): mean vs median [trace_seed] | OOD | 46.07 | 45.47 | +0.60 | [-0.07, 1.33] | 0.077 | 0.097 |

### TimesFM mean head (native, unbiased) vs decile average

| A vs B | split | RMSE A | RMSE B | Δ(A−B) | 95% CI | p_boot | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| timesfm zero-shot k=0: meanhead vs median | ID | 96.22 | 97.54 | -1.32 | [-9.35, 7.06] | 0.743 | 0.688 |
| timesfm zero-shot k=0: meanhead vs median | OOD | 83.19 | 87.70 | -4.51 | [-10.08, 7.21] | 0.349 | 0.438 |
| timesfm zero-shot k=0: meanhead vs decile-mean | ID | 96.22 | 95.88 | +0.34 | [-1.93, 3.01] | 0.799 | 0.688 |
| timesfm zero-shot k=0: meanhead vs decile-mean | OOD | 83.19 | 83.30 | -0.11 | [-2.29, 3.12] | 0.969 | 0.625 |
| timesfm best: mmr_euclid k=10: meanhead vs median | ID | 25.86 | 23.85 | +2.01 | [0.72, 4.04] | 0.013 | 0.156 |
| timesfm best: mmr_euclid k=10: meanhead vs median | OOD | 41.08 | 37.62 | +3.46 | [0.99, 5.48] | 0.019 | 0.312 |
| timesfm best: mmr_euclid k=10: meanhead vs decile-mean | ID | 25.86 | 24.91 | +0.96 | [0.35, 1.95] | 0.010 | 0.156 |
| timesfm best: mmr_euclid k=10: meanhead vs decile-mean | OOD | 41.08 | 39.19 | +1.89 | [0.67, 2.88] | 0.019 | 0.312 |
| timesfm oracle_tail k=10 (cheats): meanhead vs median | ID | 16.73 | 15.99 | +0.74 | [0.12, 1.71] | 0.031 | 0.312 |
| timesfm oracle_tail k=10 (cheats): meanhead vs median | OOD | 11.04 | 8.10 | +2.94 | [-0.53, 6.45] | 0.120 | 0.188 |
| timesfm oracle_tail k=10 (cheats): meanhead vs decile-mean | ID | 16.73 | 15.83 | +0.90 | [0.23, 1.35] | 0.030 | 0.312 |
| timesfm oracle_tail k=10 (cheats): meanhead vs decile-mean | OOD | 11.04 | 9.69 | +1.36 | [-0.48, 3.27] | 0.310 | 0.625 |
| timesfm random k=10 (20 seeds): meanhead vs median | ID | 46.24 | 47.05 | -0.82 | [-1.55, 0.06] | 0.066 | 0.156 |
| timesfm random k=10 (20 seeds): meanhead vs median [trace_seed] | ID | 46.24 | 47.05 | -0.82 | [-1.49, -0.12] | 0.024 | 0.006 |
| timesfm random k=10 (20 seeds): meanhead vs median | OOD | 54.48 | 53.85 | +0.64 | [-1.81, 2.79] | 0.715 | 0.438 |
| timesfm random k=10 (20 seeds): meanhead vs median [trace_seed] | OOD | 54.48 | 53.85 | +0.64 | [-0.49, 1.86] | 0.295 | 0.222 |
| timesfm random k=10 (20 seeds): meanhead vs decile-mean | ID | 46.24 | 46.71 | -0.48 | [-0.83, 0.05] | 0.083 | 0.219 |
| timesfm random k=10 (20 seeds): meanhead vs decile-mean [trace_seed] | ID | 46.24 | 46.71 | -0.48 | [-0.80, -0.14] | 0.007 | 0.003 |
| timesfm random k=10 (20 seeds): meanhead vs decile-mean | OOD | 54.48 | 54.55 | -0.07 | [-0.91, 0.55] | 0.880 | 0.625 |
| timesfm random k=10 (20 seeds): meanhead vs decile-mean [trace_seed] | OOD | 54.48 | 54.55 | -0.07 | [-0.62, 0.49] | 0.818 | 0.479 |

## Seed ensembling (random k=10, 20 example sets)

`per-seed` is the standard aggregation (mean over seeds of each seed's RMSE); `seed-ens` averages the 20 predicted tail means per trace before scoring (≡ averaging forecasts: the tail mean is linear). Paired rows test seed-ens vs the per-seed run over traces.

| model | decoding | per-seed RMSE (ID / OOD) | seed-ens RMSE (ID / OOD) | Δ ID | Δ OOD |
|---|---|---|---|---|---|
| tirex | median | 44.61±13.65 / 48.56±8.15 | 42.90 / 45.88 | -1.71 | -2.67 |
| tirex | mean | 43.91±13.44 / 48.49±8.12 | 42.20 / 45.79 | -1.70 | -2.69 |
| timesfm | median | 45.73±11.35 / 53.24±8.31 | 42.58 / 50.00 | -3.15 | -3.23 |
| timesfm | mean | 45.36±11.44 / 53.95±8.33 | 42.16 / 50.75 | -3.20 | -3.20 |
| timesfm | meanhead | 44.80±11.72 / 53.86±8.46 | 41.77 / 50.75 | -3.03 | -3.10 |
| chronos2 | median | 44.42±8.36 / 49.18±6.63 | 43.42 / 48.12 | -1.00 | -1.06 |
| chronos2 | mean | 39.18±8.23 / 46.07±6.35 | 37.48 / 44.25 | -1.70 | -1.81 |
| chronos_bolt | median | 38.84±12.97 / 45.16±5.41 | 36.04 / 41.71 | -2.80 | -3.45 |
| chronos_bolt | mean | 37.89±12.85 / 45.78±5.27 | 34.27 / 41.67 | -3.63 | -4.12 |

| A vs B | split | RMSE A | RMSE B | Δ(A−B) | 95% CI | p_boot | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| tirex median: seed-ens vs per-seed | ID | 42.90 | 46.55 | -3.65 | [-6.27, -2.85] | 0.000 | 0.031 |
| tirex median: seed-ens vs per-seed | OOD | 45.88 | 49.20 | -3.32 | [-10.85, -2.27] | 0.000 | 0.062 |
| tirex mean: seed-ens vs per-seed | ID | 42.20 | 45.82 | -3.61 | [-6.18, -2.82] | 0.000 | 0.031 |
| tirex mean: seed-ens vs per-seed | OOD | 45.79 | 49.13 | -3.33 | [-10.94, -2.31] | 0.000 | 0.062 |
| timesfm median: seed-ens vs per-seed | ID | 42.58 | 47.05 | -4.47 | [-8.31, -3.22] | 0.000 | 0.031 |
| timesfm median: seed-ens vs per-seed | OOD | 50.00 | 53.85 | -3.85 | [-11.28, -2.83] | 0.000 | 0.062 |
| timesfm mean: seed-ens vs per-seed | ID | 42.16 | 46.71 | -4.55 | [-8.22, -3.31] | 0.000 | 0.031 |
| timesfm mean: seed-ens vs per-seed | OOD | 50.75 | 54.55 | -3.80 | [-10.78, -2.75] | 0.000 | 0.062 |
| timesfm meanhead: seed-ens vs per-seed | ID | 41.77 | 46.24 | -4.47 | [-7.89, -3.32] | 0.000 | 0.031 |
| timesfm meanhead: seed-ens vs per-seed | OOD | 50.75 | 54.48 | -3.73 | [-10.55, -2.63] | 0.000 | 0.062 |
| chronos2 median: seed-ens vs per-seed | ID | 43.42 | 45.16 | -1.74 | [-3.86, -0.98] | 0.000 | 0.031 |
| chronos2 median: seed-ens vs per-seed | OOD | 48.12 | 49.60 | -1.48 | [-5.97, -1.02] | 0.000 | 0.062 |
| chronos2 mean: seed-ens vs per-seed | ID | 37.48 | 39.99 | -2.51 | [-5.19, -1.63] | 0.000 | 0.031 |
| chronos2 mean: seed-ens vs per-seed | OOD | 44.25 | 46.48 | -2.23 | [-6.16, -1.54] | 0.000 | 0.062 |
| chronos_bolt median: seed-ens vs per-seed | ID | 36.04 | 40.84 | -4.80 | [-9.63, -3.61] | 0.000 | 0.031 |
| chronos_bolt median: seed-ens vs per-seed | OOD | 41.71 | 45.47 | -3.76 | [-10.64, -2.58] | 0.000 | 0.062 |
| chronos_bolt mean: seed-ens vs per-seed | ID | 34.27 | 39.91 | -5.65 | [-11.39, -4.18] | 0.000 | 0.031 |
| chronos_bolt mean: seed-ens vs per-seed | OOD | 41.67 | 46.07 | -4.40 | [-10.89, -3.07] | 0.000 | 0.062 |

## Cross-model ensembling (best configs, deterministic cells)

Per-trace tail-mean averages across the models' best-config forecasts (Bolt mmr k=10, TimesFM mmr k=10, TiRex ctx_euclid k=10, Chronos-2 mmr k=5), per decoding. Compared against the best single model of that decoding (lowest ID RMSE).

### Decoding: median

| ensemble | ID | OOD | Δ ID vs best single | Δ OOD |
|---|---|---|---|---|
| tirex (single) | 30.77 | 38.62 | — | — |
| timesfm (single) | 23.85 | 37.62 | — | — |
| chronos2 (single) | 29.40 | 42.56 | — | — |
| chronos_bolt (single) ← best single | 23.28 | 37.81 | — | — |
| chronos2 + chronos_bolt | 25.34 | 39.94 | +2.06 | +2.14 |
| chronos2 + timesfm | 26.21 | 39.73 | +2.92 | +1.92 |
| chronos2 + tirex | 28.86 | 38.95 | +5.58 | +1.14 |
| chronos_bolt + timesfm | 22.97 | 37.64 | -0.31 | -0.16 |
| chronos_bolt + tirex | 26.31 | 37.54 | +3.03 | -0.27 |
| timesfm + tirex | 26.74 | 37.23 | +3.46 | -0.58 |
| chronos2 + chronos_bolt + timesfm + tirex | 25.67 | 38.18 | +2.39 | +0.37 |

| A vs B | split | RMSE A | RMSE B | Δ(A−B) | 95% CI | p_boot | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| [median] chronos2+chronos_bolt vs chronos_bolt | ID | 25.34 | 23.28 | +2.06 | [-4.74, 6.67] | 0.446 | 0.562 |
| [median] chronos2+chronos_bolt vs chronos_bolt | OOD | 39.94 | 37.81 | +2.14 | [-3.15, 4.56] | 0.312 | 0.438 |
| [median] chronos2+timesfm vs chronos_bolt | ID | 26.21 | 23.28 | +2.92 | [-7.80, 8.31] | 0.465 | 0.438 |
| [median] chronos2+timesfm vs chronos_bolt | OOD | 39.73 | 37.81 | +1.92 | [-3.55, 5.26] | 0.378 | 0.312 |
| [median] chronos2+tirex vs chronos_bolt | ID | 28.86 | 23.28 | +5.58 | [-3.17, 12.93] | 0.163 | 0.219 |
| [median] chronos2+tirex vs chronos_bolt | OOD | 38.95 | 37.81 | +1.14 | [-1.80, 4.82] | 0.362 | 0.438 |
| [median] chronos_bolt+timesfm vs chronos_bolt | ID | 22.97 | 23.28 | -0.31 | [-6.38, 3.29] | 0.899 | 0.688 |
| [median] chronos_bolt+timesfm vs chronos_bolt | OOD | 37.64 | 37.81 | -0.16 | [-1.54, 2.26] | 0.794 | 0.812 |
| [median] chronos_bolt+tirex vs chronos_bolt | ID | 26.31 | 23.28 | +3.03 | [-1.21, 6.85] | 0.162 | 0.562 |
| [median] chronos_bolt+tirex vs chronos_bolt | OOD | 37.54 | 37.81 | -0.27 | [-5.55, 7.80] | 0.847 | 1.000 |
| [median] timesfm+tirex vs chronos_bolt | ID | 26.74 | 23.28 | +3.46 | [-6.12, 10.01] | 0.455 | 0.844 |
| [median] timesfm+tirex vs chronos_bolt | OOD | 37.23 | 37.81 | -0.58 | [-5.23, 7.73] | 0.844 | 0.625 |
| [median] chronos2+chronos_bolt+timesfm+tirex vs chronos_bolt | ID | 25.67 | 23.28 | +2.39 | [-4.67, 7.28] | 0.439 | 0.438 |
| [median] chronos2+chronos_bolt+timesfm+tirex vs chronos_bolt | OOD | 38.18 | 37.81 | +0.37 | [-0.39, 1.97] | 0.492 | 0.812 |

### Decoding: mean

| ensemble | ID | OOD | Δ ID vs best single | Δ OOD |
|---|---|---|---|---|
| tirex (single) | 30.68 | 37.95 | — | — |
| timesfm (single) | 24.91 | 39.19 | — | — |
| chronos2 (single) | 27.06 | 42.64 | — | — |
| chronos_bolt (single) ← best single | 22.63 | 38.68 | — | — |
| chronos2 + chronos_bolt | 23.71 | 40.42 | +1.08 | +1.74 |
| chronos2 + timesfm | 25.41 | 40.36 | +2.78 | +1.68 |
| chronos2 + tirex | 26.58 | 38.93 | +3.95 | +0.24 |
| chronos_bolt + timesfm | 23.05 | 38.78 | +0.42 | +0.10 |
| chronos_bolt + tirex | 25.84 | 37.73 | +3.20 | -0.95 |
| timesfm + tirex | 27.08 | 37.59 | +4.44 | -1.09 |
| chronos2 + chronos_bolt + timesfm + tirex | 24.72 | 38.65 | +2.08 | -0.03 |

| A vs B | split | RMSE A | RMSE B | Δ(A−B) | 95% CI | p_boot | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| [mean] chronos2+chronos_bolt vs chronos_bolt | ID | 23.71 | 22.63 | +1.08 | [-5.74, 5.42] | 0.576 | 0.438 |
| [mean] chronos2+chronos_bolt vs chronos_bolt | OOD | 40.42 | 38.68 | +1.74 | [-3.81, 4.62] | 0.439 | 0.438 |
| [mean] chronos2+timesfm vs chronos_bolt | ID | 25.41 | 22.63 | +2.78 | [-7.08, 7.15] | 0.418 | 0.312 |
| [mean] chronos2+timesfm vs chronos_bolt | OOD | 40.36 | 38.68 | +1.68 | [-4.11, 4.72] | 0.409 | 0.625 |
| [mean] chronos2+tirex vs chronos_bolt | ID | 26.58 | 22.63 | +3.95 | [-3.86, 9.86] | 0.259 | 0.312 |
| [mean] chronos2+tirex vs chronos_bolt | OOD | 38.93 | 38.68 | +0.24 | [-3.63, 3.47] | 0.766 | 0.625 |
| [mean] chronos_bolt+timesfm vs chronos_bolt | ID | 23.05 | 22.63 | +0.42 | [-5.92, 3.69] | 0.841 | 0.844 |
| [mean] chronos_bolt+timesfm vs chronos_bolt | OOD | 38.78 | 38.68 | +0.10 | [-2.04, 3.31] | 0.909 | 0.625 |
| [mean] chronos_bolt+tirex vs chronos_bolt | ID | 25.84 | 22.63 | +3.20 | [-0.14, 6.84] | 0.063 | 0.438 |
| [mean] chronos_bolt+tirex vs chronos_bolt | OOD | 37.73 | 38.68 | -0.95 | [-6.11, 6.18] | 0.725 | 0.812 |
| [mean] timesfm+tirex vs chronos_bolt | ID | 27.08 | 22.63 | +4.44 | [-4.13, 11.48] | 0.246 | 0.562 |
| [mean] timesfm+tirex vs chronos_bolt | OOD | 37.59 | 38.68 | -1.09 | [-5.28, 6.02] | 0.710 | 0.625 |
| [mean] chronos2+chronos_bolt+timesfm+tirex vs chronos_bolt | ID | 24.72 | 22.63 | +2.08 | [-5.18, 6.47] | 0.472 | 0.438 |
| [mean] chronos2+chronos_bolt+timesfm+tirex vs chronos_bolt | OOD | 38.65 | 38.68 | -0.03 | [-0.81, 0.96] | 0.907 | 0.812 |

## Decision — does mean decoding become the default?

Across the 16 (model, config) cells with both decodings: mean improves ID RMSE in 14, worsens it in 2 (bootstrap-significant cells marked •):

| model | config | Δ ID (mean−median) | significant |
|---|---|---|---|
| chronos2 | zero-shot k=0 | -20.40 | • |
| chronos2 | random k=10 (20 seeds) | -5.24 | • |
| tirex | zero-shot k=0 | -3.25 | • |
| chronos_bolt | zero-shot k=0 | -2.62 | • |
| chronos2 | best: mmr_euclid k=5 | -2.34 |  |
| chronos2 | oracle_tail k=10 (cheats) | -1.70 | • |
| timesfm | zero-shot k=0 | -1.66 |  |
| chronos_bolt | oracle_tail k=10 (cheats) | -1.28 |  |
| chronos_bolt | random k=10 (20 seeds) | -0.95 |  |
| tirex | random k=10 (20 seeds) | -0.70 | • |
| chronos_bolt | best: mmr_euclid k=10 | -0.65 |  |
| timesfm | random k=10 (20 seeds) | -0.37 |  |
| timesfm | oracle_tail k=10 (cheats) | -0.16 |  |
| tirex | best: ctx_euclid k=10 | -0.09 |  |
| tirex | oracle_tail k=10 (cheats) | +0.35 |  |
| timesfm | best: mmr_euclid k=10 | +1.05 | • |

**Verdict — adopt mean decoding per model, not globally.** For
**Chronos-2** mean decoding is uniformly better (zero-shot −20.4 ID,
random −5.2 ID significant, oracle −1.7 ID significant, best config
−2.3 ID) — it becomes the DEFAULT for all later Chronos-2 work,
including Phase 5's finetuned-Chronos-2 ICL runs. For **Chronos-Bolt**
and **TiRex** mean is a small free win (ID never worse, the only
regression is TiRex's cheating oracle at +0.35 n.s.); adopt it — it
produces the new best legitimate training-free cell, Bolt mmr_euclid
shared k=10 at **22.63 ID** (was 23.28). For **TimesFM** keep the
median: at its best config both the decile mean (+1.05 ID / +1.57 OOD)
and its own native mean head (+2.01 ID / +3.46 OOD) are
bootstrap-significantly WORSE — and since meanhead ≥ decile-mean ≥
median there, the failure is not the decile truncation bias but the
mean statistic itself interacting badly with TimesFM's wide right tail
under ICL. The pattern across models: the worse the level calibration
of a config, the more the mean helps (anchors ≫ random ≫ best), exactly
what a skew correction should do once shared scaling has already moved
the level most of the way.

**Ensembling.** Seed ensembling (average the 20 random-example-set
forecasts before scoring) is significantly better than per-seed scoring
for every model and decoding (ID −1.7 to −5.7, OOD −1.5 to −4.4, all
bootstrap p ≤ 0.001) — a real variance reduction, but ensembled random
selection (best: Bolt mean 34.27 ID) still loses clearly to plain
retrieval (22.63), so it is a fallback when retrieval is unavailable,
not a replacement. Cross-model ensembling of the best configs never
beats the best single model ID (closest: Bolt+TimesFM −0.31 n.s.; the
TODO's literal Bolt+TiRex is +3.0 WORSE) — the models' tail-level
errors are too correlated for averaging to pay.

## Interpretation notes

- **Why mean decoding at all.** The metric is tail RMSE of a positive,
  right-skewed flux; the RMSE-optimal point forecast is the conditional
  MEAN, and the median of a right-skewed predictive distribution sits
  below it. Phase 3's shared scaling already fixed the *level transfer*
  problem, so the remaining median-vs-mean gap was expected to be small —
  the deliverable is the decision either way.
- **The decile average is not the exact mean.** (1/9)·Σ q₀.₁..q₀.₉
  truncates the predictive distribution beyond q0.1/q0.9, so it
  UNDERestimates the mean of a right-skewed distribution (smoke D3:
  lognormal σ=1 has median 1.00 < decile-avg 1.33 < exact mean 1.65).
  TimesFM's native mean head (`meanhead`) is the unbiased-head
  cross-check for this bias.
- **Decoding feeds back through the rollout.** The decoded point is
  appended to the context and re-normalized each 64-step iteration, so
  mean decoding changes the whole trajectory, not just the final
  read-out; the k=0 anchors isolate this pure decoding+feedback effect.
- **TiRex has NO native mean.** The library's second ``forecast()``
  return is labeled mean but selects q0.5 by index
  (``tirex/models/tirex.py``: ``# median as mean``) — confirmed
  bit-identical to the median at runtime by smoke D1. The TODO's "TiRex
  returns the mean natively" was wrong.
- **TimesFM layout evidence** (smoke D2): the full forecast's last dim is
  ``[mean, q0.1..q0.9]``; the first ``forecast()`` return is literally
  ``full[..., 5]`` (the median; bit-equal), indices 1..9 are monotone,
  and index 0 is a distinct head sitting above the median on a real flux
  trace (+0.17 in normalized space) — consistent with right skew.
- **Ensembling is post-processing.** The recorded per-seed per-trace
  ``pred_tail_mean`` is linear in the forecast, so averaging tail means ≡
  averaging forecasts before scoring; no harness change. Seed ensembling
  removes example-selection variance from the random cells; cross-model
  ensembling averages the deterministic best-config cells.
- MPS is not bit-deterministic across process runs: headline pairs live
  within the single v5 grid run; the v5 median cells double as a
  cross-run reproduction of the v3/v4 twins (reported, not asserted).
- Wilcoxon p floors at 0.031 (n=6 ID) / 0.0625 (n=5 OOD) under
  pairing="trace"; the bootstrap CI is the primary evidence.
