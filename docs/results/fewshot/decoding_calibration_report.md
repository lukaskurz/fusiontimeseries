# Decoding calibration on training traces

Calibration set: 244 non-benchmark pool traces, leave-one-out retrieval, same 80-step context / tail-80 protocol as the benchmark.


## chronos2

### zeroshot k=0

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 77.42 | -61.14 | 0.07 | 109.91 | 85.86 |
| q0.60  | 68.06 | -41.69 | 0.15 | 100.23 | 73.13 |
| q0.70  | 64.50 | -19.89 | 0.24 | 83.61 | 67.04 |
| q0.80 **bias0** **cov50** _test-argmin_ | 79.98 | +15.63 | 0.42 | 67.23 | 80.13 |
| q0.90  | 146.64 | +94.81 | 0.76 | 95.17 | 111.18 |
| q0.95  | 265.88 | +210.49 | 0.93 | 163.68 | 215.12 |
| q0.99  | 1369.87 | +1092.41 | 1.00 | 758.52 | 1686.09 |
| mean **argmin** | 58.18 | -37.26 | 0.13 | 89.51 | 67.94 |

**5-fold selection** (stratified by tail level) picks: mean, mean, mean, mean, mean → unanimous
Held-out calibration RMSE: median 77.42 | mean 58.18 | CV-selected q 58.18

**Transfer ID** (baselines: median 109.91, mean 89.51; test-argmin q0.80 → 67.23 = unreachable upper bound)
  - argmin → mean →  89.51 | vs median -20.40 [-30.22, -12.54] | vs mean  +0.00 [+0.00, +0.00]
  - bias0  → q0.80 →  67.23 | vs median -42.67 [-73.69, -18.30] | vs mean -22.28 [-44.11, -5.49]
  - cov50  → q0.80 →  67.23 | vs median -42.67 [-73.69, -18.30] | vs mean -22.28 [-44.11, -5.49]
**Transfer OOD** (baselines: median 85.86, mean 67.94; test-argmin q0.80 → 80.13 = unreachable upper bound)
  - argmin → mean →  67.94 | vs median -17.92 [-23.99, -12.96] | vs mean  +0.00 [+0.00, +0.00]
  - bias0  → q0.80 →  80.13 | vs median  -5.73 [-43.52, +41.69] | vs mean +12.19 [-19.78, +58.01]
  - cov50  → q0.80 →  80.13 | vs median  -5.73 [-43.52, +41.69] | vs mean +12.19 [-19.78, +58.01]

### mmr_euclid k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 29.35 | -13.61 | 0.33 | 29.40 | 42.56 |
| q0.60 **argmin** **bias0** **cov50** _test-argmin_ | 25.93 | -3.17 | 0.52 | 21.28 | 40.00 |
| q0.70  | 28.32 | +9.37 | 0.68 | 25.04 | 41.48 |
| q0.80  | 40.36 | +27.20 | 0.84 | 39.80 | 49.27 |
| q0.90  | 67.77 | +57.27 | 0.96 | 68.54 | 64.47 |
| q0.95  | 102.78 | +92.50 | 0.98 | 109.44 | 87.03 |
| q0.99  | 213.15 | +199.62 | 0.99 | 226.08 | 162.19 |
| mean  | 27.96 | -7.46 | 0.43 | 27.06 | 42.64 |

**5-fold selection** (stratified by tail level) picks: q0.60, q0.60, q0.60, q0.60, q0.60 → unanimous
Held-out calibration RMSE: median 29.35 | mean 27.96 | CV-selected q 25.93

**Transfer ID** (baselines: median 29.40, mean 27.06; test-argmin q0.60 → 21.28 = unreachable upper bound)
  - argmin → q0.60 →  21.28 | vs median  -8.12 [-14.83, -1.54] | vs mean  -5.79 [-9.93, -1.45]
  - bias0  → q0.60 →  21.28 | vs median  -8.12 [-14.83, -1.54] | vs mean  -5.79 [-9.93, -1.45]
  - cov50  → q0.60 →  21.28 | vs median  -8.12 [-14.83, -1.54] | vs mean  -5.79 [-9.93, -1.45]
**Transfer OOD** (baselines: median 42.56, mean 42.64; test-argmin q0.60 → 40.00 = unreachable upper bound)
  - argmin → q0.60 →  40.00 | vs median  -2.56 [-9.36, +9.42] | vs mean  -2.63 [-6.99, +5.42]
  - bias0  → q0.60 →  40.00 | vs median  -2.56 [-9.36, +9.42] | vs mean  -2.63 [-6.99, +5.42]
  - cov50  → q0.60 →  40.00 | vs median  -2.56 [-9.36, +9.42] | vs mean  -2.63 [-6.99, +5.42]

### ctx_level k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 25.07 | -10.51 | 0.34 | 43.14 | 30.27 |
| q0.60 **argmin** **bias0** **cov50** | 23.32 | -2.11 | 0.50 | 37.11 | 27.79 |
| q0.70  | 26.05 | +8.05 | 0.67 | 30.23 | 30.46 |
| q0.80 _test-argmin_ | 35.68 | +22.03 | 0.82 | 23.23 | 35.50 |
| q0.90  | 57.22 | +45.12 | 0.93 | 24.55 | 47.69 |
| q0.95  | 85.87 | +72.92 | 0.97 | 43.08 | 74.84 |
| q0.99  | 175.94 | +155.49 | 1.00 | 105.15 | 184.25 |
| mean  | 23.85 | -6.55 | 0.42 | 40.56 | 31.31 |

**5-fold selection** (stratified by tail level) picks: q0.60, q0.60, q0.60, q0.60, q0.60 → unanimous
Held-out calibration RMSE: median 25.07 | mean 23.85 | CV-selected q 23.32

**Transfer ID** (baselines: median 43.14, mean 40.56; test-argmin q0.80 → 23.23 = unreachable upper bound)
  - argmin → q0.60 →  37.11 | vs median  -6.02 [-8.33, -4.09] | vs mean  -3.45 [-4.62, -2.23]
  - bias0  → q0.60 →  37.11 | vs median  -6.02 [-8.33, -4.09] | vs mean  -3.45 [-4.62, -2.23]
  - cov50  → q0.60 →  37.11 | vs median  -6.02 [-8.33, -4.09] | vs mean  -3.45 [-4.62, -2.23]
**Transfer OOD** (baselines: median 30.27, mean 31.31; test-argmin q0.80 → 35.50 = unreachable upper bound)
  - argmin → q0.60 →  27.79 | vs median  -2.48 [-6.80, +8.23] | vs mean  -3.52 [-7.09, +4.12]
  - bias0  → q0.60 →  27.79 | vs median  -2.48 [-6.80, +8.23] | vs mean  -3.52 [-7.09, +4.12]
  - cov50  → q0.60 →  27.79 | vs median  -2.48 [-6.80, +8.23] | vs mean  -3.52 [-7.09, +4.12]


## chronos_bolt

### zeroshot k=0

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 68.44 | -56.92 | 0.05 | 111.65 | 90.69 |
| q0.60 **argmin** | 61.72 | -41.58 | 0.11 | 103.88 | 84.27 |
| q0.70  | 62.98 | -21.83 | 0.24 | 94.17 | 83.07 |
| q0.80 **bias0** **cov50** _test-argmin_ | 80.74 | +9.24 | 0.41 | 87.48 | 88.70 |
| q0.90  | 145.40 | +77.81 | 0.66 | 91.70 | 114.32 |
| mean  | 66.70 | -54.99 | 0.05 | 109.03 | 92.32 |

**5-fold selection** (stratified by tail level) picks: q0.70, q0.60, q0.60, q0.60, q0.60 → SPLIT
Held-out calibration RMSE: median 68.44 | mean 66.70 | CV-selected q 63.52

**Transfer ID** (baselines: median 111.65, mean 109.03; test-argmin q0.80 → 87.48 = unreachable upper bound)
  - argmin → q0.60 → 103.88 | vs median  -7.77 [-16.04, -1.80] | vs mean  -5.15 [-12.05, -0.57]
  - bias0  → q0.80 →  87.48 | vs median -24.17 [-55.77, -6.08] | vs mean -21.55 [-51.68, -4.95]
  - cov50  → q0.80 →  87.48 | vs median -24.17 [-55.77, -6.08] | vs mean -21.55 [-51.68, -4.95]
**Transfer OOD** (baselines: median 90.69, mean 92.32; test-argmin q0.80 → 88.70 = unreachable upper bound)
  - argmin → q0.60 →  84.27 | vs median  -6.42 [-16.62, -2.19] | vs mean  -8.05 [-23.07, -1.22]
  - bias0  → q0.80 →  88.70 | vs median  -1.99 [-29.35, +38.45] | vs mean  -3.62 [-34.73, +37.90]
  - cov50  → q0.80 →  88.70 | vs median  -1.99 [-29.35, +38.45] | vs mean  -3.62 [-34.73, +37.90]

### mmr_euclid k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50 **argmin** | 28.99 | -4.47 | 0.43 | 26.35 | 37.20 |
| q0.60 _test-argmin_ | 32.12 | +12.08 | 0.66 | 21.74 | 35.21 |
| q0.70  | 45.33 | +31.80 | 0.88 | 32.62 | 43.51 |
| q0.80  | 69.57 | +58.59 | 0.95 | 55.30 | 62.61 |
| q0.90  | 120.49 | +109.12 | 1.00 | 97.46 | 99.21 |
| mean **bias0** **cov50** | 29.01 | -2.61 | 0.46 | 25.86 | 38.05 |

**5-fold selection** (stratified by tail level) picks: mean, q0.50, q0.50, q0.50, q0.50 → SPLIT
Held-out calibration RMSE: median 28.99 | mean 29.01 | CV-selected q 29.07

**Transfer ID** (baselines: median 26.35, mean 25.86; test-argmin q0.60 → 21.74 = unreachable upper bound)
  - argmin → q0.50 →  26.35 | vs median  +0.00 [+0.00, +0.00] | vs mean  +0.49 [-1.28, +2.65]
  - bias0  → mean →  25.86 | vs median  -0.49 [-2.65, +1.28] | vs mean  +0.00 [+0.00, +0.00]
  - cov50  → mean →  25.86 | vs median  -0.49 [-2.65, +1.28] | vs mean  +0.00 [+0.00, +0.00]
**Transfer OOD** (baselines: median 37.20, mean 38.05; test-argmin q0.60 → 35.21 = unreachable upper bound)
  - argmin → q0.50 →  37.20 | vs median  +0.00 [+0.00, +0.00] | vs mean  -0.84 [-2.37, -0.32]
  - bias0  → mean →  38.05 | vs median  +0.84 [+0.32, +2.37] | vs mean  +0.00 [+0.00, +0.00]
  - cov50  → mean →  38.05 | vs median  +0.84 [+0.32, +2.37] | vs mean  +0.00 [+0.00, +0.00]

### ctx_level k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50 **argmin** **bias0** **cov50** | 24.00 | +1.99 | 0.52 | 39.29 | 22.45 |
| q0.60  | 29.44 | +14.18 | 0.74 | 31.37 | 26.63 |
| q0.70 _test-argmin_ | 40.73 | +28.28 | 0.85 | 27.72 | 38.11 |
| q0.80  | 59.62 | +47.76 | 0.93 | 36.40 | 56.62 |
| q0.90  | 99.37 | +85.67 | 0.99 | 75.16 | 98.02 |
| mean  | 24.19 | +2.90 | 0.54 | 38.69 | 21.91 |

**5-fold selection** (stratified by tail level) picks: q0.50, q0.50, q0.50, q0.50, q0.50 → unanimous
Held-out calibration RMSE: median 24.00 | mean 24.19 | CV-selected q 24.00

**Transfer ID** (baselines: median 39.29, mean 38.69; test-argmin q0.70 → 27.72 = unreachable upper bound)
  - argmin → q0.50 →  39.29 | vs median  +0.00 [+0.00, +0.00] | vs mean  +0.60 [-0.19, +1.12]
  - bias0  → q0.50 →  39.29 | vs median  +0.00 [+0.00, +0.00] | vs mean  +0.60 [-0.19, +1.12]
  - cov50  → q0.50 →  39.29 | vs median  +0.00 [+0.00, +0.00] | vs mean  +0.60 [-0.19, +1.12]
**Transfer OOD** (baselines: median 22.45, mean 21.91; test-argmin q0.70 → 38.11 = unreachable upper bound)
  - argmin → q0.50 →  22.45 | vs median  +0.00 [+0.00, +0.00] | vs mean  +0.54 [-1.02, +1.55]
  - bias0  → q0.50 →  22.45 | vs median  +0.00 [+0.00, +0.00] | vs mean  +0.54 [-1.02, +1.55]
  - cov50  → q0.50 →  22.45 | vs median  +0.00 [+0.00, +0.00] | vs mean  +0.54 [-1.02, +1.55]


## tirex

### zeroshot k=0

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 50.39 | -41.10 | 0.05 | 78.92 | 62.49 |
| q0.60  | 36.95 | -23.78 | 0.14 | 62.47 | 49.99 |
| q0.70 **argmin** **bias0** **cov50** | 31.26 | -5.31 | 0.34 | 45.70 | 46.06 |
| q0.80 _test-argmin_ | 50.61 | +26.28 | 0.73 | 32.77 | 54.80 |
| q0.90  | 112.73 | +89.73 | 0.95 | 64.06 | 94.75 |
| mean  | 47.35 | -37.60 | 0.07 | 75.67 | 59.99 |

**5-fold selection** (stratified by tail level) picks: q0.70, q0.70, q0.70, q0.70, q0.70 → unanimous
Held-out calibration RMSE: median 50.39 | mean 47.35 | CV-selected q 31.26

**Transfer ID** (baselines: median 78.92, mean 75.67; test-argmin q0.80 → 32.77 = unreachable upper bound)
  - argmin → q0.70 →  45.70 | vs median -33.22 [-38.39, -29.59] | vs mean -29.97 [-34.55, -26.60]
  - bias0  → q0.70 →  45.70 | vs median -33.22 [-38.39, -29.59] | vs mean -29.97 [-34.55, -26.60]
  - cov50  → q0.70 →  45.70 | vs median -33.22 [-38.39, -29.59] | vs mean -29.97 [-34.55, -26.60]
**Transfer OOD** (baselines: median 62.49, mean 59.99; test-argmin q0.80 → 54.80 = unreachable upper bound)
  - argmin → q0.70 →  46.06 | vs median -16.43 [-43.13, +12.38] | vs mean -13.93 [-38.38, +16.13]
  - bias0  → q0.70 →  46.06 | vs median -16.43 [-43.13, +12.38] | vs mean -13.93 [-38.38, +16.13]
  - cov50  → q0.70 →  46.06 | vs median -16.43 [-43.13, +12.38] | vs mean -13.93 [-38.38, +16.13]

### mmr_euclid k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50 **cov50** | 26.13 | -6.57 | 0.46 | 35.51 | 32.38 |
| q0.60 **bias0** | 26.50 | +4.64 | 0.64 | 25.25 | 30.00 |
| q0.70 _test-argmin_ | 33.30 | +17.82 | 0.76 | 20.63 | 33.10 |
| q0.80  | 50.44 | +38.15 | 0.90 | 34.08 | 45.16 |
| q0.90  | 81.71 | +70.19 | 0.98 | 68.27 | 72.35 |
| mean **argmin** | 25.91 | -6.08 | 0.46 | 36.50 | 33.18 |

**5-fold selection** (stratified by tail level) picks: mean, mean, mean, mean, mean → unanimous
Held-out calibration RMSE: median 26.13 | mean 25.91 | CV-selected q 25.91

**Transfer ID** (baselines: median 35.51, mean 36.50; test-argmin q0.70 → 20.63 = unreachable upper bound)
  - argmin → mean →  36.50 | vs median  +0.98 [-0.98, +2.81] | vs mean  +0.00 [+0.00, +0.00]
  - bias0  → q0.60 →  25.25 | vs median -10.27 [-16.49, -5.72] | vs mean -11.25 [-19.16, -4.97]
  - cov50  → q0.50 →  35.51 | vs median  +0.00 [+0.00, +0.00] | vs mean  -0.98 [-2.81, +0.98]
**Transfer OOD** (baselines: median 32.38, mean 33.18; test-argmin q0.70 → 33.10 = unreachable upper bound)
  - argmin → mean →  33.18 | vs median  +0.80 [-0.32, +1.83] | vs mean  +0.00 [+0.00, +0.00]
  - bias0  → q0.60 →  30.00 | vs median  -2.39 [-9.78, +9.05] | vs mean  -3.19 [-11.31, +9.37]
  - cov50  → q0.50 →  32.38 | vs median  +0.00 [+0.00, +0.00] | vs mean  -0.80 [-1.83, +0.32]

### ctx_level k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 22.52 | -7.17 | 0.41 | 40.38 | 29.29 |
| q0.60 **argmin** **bias0** **cov50** | 21.33 | +1.44 | 0.56 | 34.08 | 26.89 |
| q0.70  | 25.18 | +11.44 | 0.74 | 27.41 | 27.87 |
| q0.80 _test-argmin_ | 38.01 | +27.21 | 0.86 | 21.63 | 36.64 |
| q0.90  | 63.95 | +52.76 | 0.95 | 33.15 | 57.57 |
| mean  | 22.47 | -6.62 | 0.42 | 39.33 | 29.50 |

**5-fold selection** (stratified by tail level) picks: q0.60, q0.60, q0.60, q0.60, q0.60 → unanimous
Held-out calibration RMSE: median 22.52 | mean 22.47 | CV-selected q 21.33

**Transfer ID** (baselines: median 40.38, mean 39.33; test-argmin q0.80 → 21.63 = unreachable upper bound)
  - argmin → q0.60 →  34.08 | vs median  -6.30 [-8.59, -4.26] | vs mean  -5.25 [-7.78, -3.44]
  - bias0  → q0.60 →  34.08 | vs median  -6.30 [-8.59, -4.26] | vs mean  -5.25 [-7.78, -3.44]
  - cov50  → q0.60 →  34.08 | vs median  -6.30 [-8.59, -4.26] | vs mean  -5.25 [-7.78, -3.44]
**Transfer OOD** (baselines: median 29.29, mean 29.50; test-argmin q0.80 → 36.64 = unreachable upper bound)
  - argmin → q0.60 →  26.89 | vs median  -2.40 [-6.81, +7.67] | vs mean  -2.61 [-7.11, +7.32]
  - bias0  → q0.60 →  26.89 | vs median  -2.40 [-6.81, +7.67] | vs mean  -2.61 [-7.11, +7.32]
  - cov50  → q0.60 →  26.89 | vs median  -2.40 [-6.81, +7.67] | vs mean  -2.61 [-7.11, +7.32]


## timesfm

### zeroshot k=0

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 68.26 | -59.63 | 0.01 | 97.54 | 87.70 |
| q0.60  | 56.32 | -45.80 | 0.04 | 84.12 | 77.54 |
| q0.70  | 48.12 | -33.99 | 0.11 | 85.02 | 70.67 |
| q0.80 **argmin** **bias0** | 38.13 | -9.90 | 0.33 | 66.86 | 64.35 |
| q0.90 **cov50** _test-argmin_ | 55.73 | +24.37 | 0.64 | 44.44 | 78.83 |
| mean  | 65.98 | -57.94 | 0.01 | 95.88 | 83.30 |
| meanhead  | 67.69 | -60.46 | 0.00 | 96.22 | 83.19 |

**5-fold selection** (stratified by tail level) picks: q0.80, q0.80, q0.80, q0.80, q0.80 → unanimous
Held-out calibration RMSE: median 68.26 | mean 65.98 | CV-selected q 38.13

**Transfer ID** (baselines: median 97.54, mean 95.88; test-argmin q0.90 → 44.44 = unreachable upper bound)
  - argmin → q0.80 →  66.86 | vs median -30.69 [-48.40, -19.50] | vs mean -29.03 [-40.89, -20.98]
  - bias0  → q0.80 →  66.86 | vs median -30.69 [-48.40, -19.50] | vs mean -29.03 [-40.89, -20.98]
  - cov50  → q0.90 →  44.44 | vs median -53.10 [-72.98, -34.69] | vs mean -51.44 [-66.06, -35.36]
**Transfer OOD** (baselines: median 87.70, mean 83.30; test-argmin q0.90 → 78.83 = unreachable upper bound)
  - argmin → q0.80 →  64.35 | vs median -23.36 [-52.93, +8.43] | vs mean -18.96 [-48.32, +12.27]
  - bias0  → q0.80 →  64.35 | vs median -23.36 [-52.93, +8.43] | vs mean -18.96 [-48.32, +12.27]
  - cov50  → q0.90 →  78.83 | vs median  -8.87 [-68.48, +73.58] | vs mean  -4.47 [-61.43, +73.21]

### mmr_euclid k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 29.58 | -8.73 | 0.40 | 28.46 | 44.31 |
| q0.60 **argmin** **bias0** _test-argmin_ | 29.13 | +6.23 | 0.62 | 21.03 | 40.95 |
| q0.70  | 36.00 | +20.48 | 0.78 | 23.70 | 43.72 |
| q0.80  | 58.79 | +48.16 | 0.94 | 48.33 | 59.48 |
| q0.90  | 94.72 | +85.22 | 0.98 | 81.66 | 78.32 |
| mean **cov50** | 29.88 | -8.81 | 0.42 | 28.46 | 44.99 |
| meanhead  | 30.60 | -9.84 | 0.41 | 29.45 | 46.72 |

**5-fold selection** (stratified by tail level) picks: q0.60, q0.60, q0.50, q0.60, q0.60 → SPLIT
Held-out calibration RMSE: median 29.58 | mean 29.88 | CV-selected q 30.16

**Transfer ID** (baselines: median 28.46, mean 28.46; test-argmin q0.60 → 21.03 = unreachable upper bound)
  - argmin → q0.60 →  21.03 | vs median  -7.42 [-15.54, +2.17] | vs mean  -7.43 [-15.27, +1.96]
  - bias0  → q0.60 →  21.03 | vs median  -7.42 [-15.54, +2.17] | vs mean  -7.43 [-15.27, +1.96]
  - cov50  → mean →  28.46 | vs median  +0.00 [-1.65, +1.26] | vs mean  +0.00 [+0.00, +0.00]
**Transfer OOD** (baselines: median 44.31, mean 44.99; test-argmin q0.60 → 40.95 = unreachable upper bound)
  - argmin → q0.60 →  40.95 | vs median  -3.36 [-18.05, +12.93] | vs mean  -4.04 [-18.03, +13.19]
  - bias0  → q0.60 →  40.95 | vs median  -3.36 [-18.05, +12.93] | vs mean  -4.04 [-18.03, +13.19]
  - cov50  → mean →  44.99 | vs median  +0.68 [-0.86, +2.47] | vs mean  +0.00 [+0.00, +0.00]

### ctx_level k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50  | 23.72 | -9.88 | 0.32 | 42.53 | 26.04 |
| q0.60 **argmin** **bias0** **cov50** | 21.68 | +1.11 | 0.57 | 34.63 | 23.70 |
| q0.70  | 25.77 | +11.75 | 0.73 | 27.88 | 26.24 |
| q0.80 _test-argmin_ | 41.17 | +30.84 | 0.91 | 26.26 | 39.47 |
| q0.90  | 69.12 | +58.80 | 0.98 | 42.09 | 63.30 |
| mean  | 23.99 | -10.54 | 0.32 | 42.41 | 27.60 |
| meanhead  | 24.15 | -11.15 | 0.32 | 43.42 | 28.56 |

**5-fold selection** (stratified by tail level) picks: q0.60, q0.60, q0.60, q0.60, q0.60 → unanimous
Held-out calibration RMSE: median 23.72 | mean 23.99 | CV-selected q 21.68

**Transfer ID** (baselines: median 42.53, mean 42.41; test-argmin q0.80 → 26.26 = unreachable upper bound)
  - argmin → q0.60 →  34.63 | vs median  -7.91 [-12.42, -5.17] | vs mean  -7.78 [-13.14, -4.70]
  - bias0  → q0.60 →  34.63 | vs median  -7.91 [-12.42, -5.17] | vs mean  -7.78 [-13.14, -4.70]
  - cov50  → q0.60 →  34.63 | vs median  -7.91 [-12.42, -5.17] | vs mean  -7.78 [-13.14, -4.70]
**Transfer OOD** (baselines: median 26.04, mean 27.60; test-argmin q0.80 → 39.47 = unreachable upper bound)
  - argmin → q0.60 →  23.70 | vs median  -2.34 [-8.56, +11.50] | vs mean  -3.91 [-11.38, +12.13]
  - bias0  → q0.60 →  23.70 | vs median  -2.34 [-8.56, +11.50] | vs mean  -3.91 [-11.38, +12.13]
  - cov50  → q0.60 →  23.70 | vs median  -2.34 [-8.56, +11.50] | vs mean  -3.91 [-11.38, +12.13]


## ft_chronos2

### zeroshot k=0

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50 **argmin** **bias0** **cov50** | 22.03 | +7.05 | 0.68 | 25.33 | 33.03 |
| q0.60  | 26.21 | +14.42 | 0.80 | 19.99 | 34.47 |
| q0.70 _test-argmin_ | 33.25 | +23.20 | 0.87 | 16.00 | 37.68 |
| q0.80  | 45.00 | +35.55 | 0.93 | 17.29 | 44.33 |
| q0.90  | 69.26 | +58.66 | 0.98 | 33.34 | 60.91 |
| q0.95  | 102.84 | +89.23 | 1.00 | 59.79 | 88.42 |
| q0.99  | 277.69 | +244.63 | 1.00 | 195.10 | 307.64 |
| mean  | 24.65 | +12.35 | 0.76 | 22.20 | 34.10 |

**5-fold selection** (stratified by tail level) picks: q0.50, q0.50, q0.50, q0.50, q0.50 → unanimous
Held-out calibration RMSE: median 22.03 | mean 24.65 | CV-selected q 22.03

**Transfer ID** (baselines: median 25.33, mean 22.20; test-argmin q0.70 → 16.00 = unreachable upper bound)
  - argmin → q0.50 →  25.33 | vs median  +0.00 [+0.00, +0.00] | vs mean  +3.13 [+2.02, +4.49]
  - bias0  → q0.50 →  25.33 | vs median  +0.00 [+0.00, +0.00] | vs mean  +3.13 [+2.02, +4.49]
  - cov50  → q0.50 →  25.33 | vs median  +0.00 [+0.00, +0.00] | vs mean  +3.13 [+2.02, +4.49]
**Transfer OOD** (baselines: median 33.03, mean 34.10; test-argmin q0.70 → 37.68 = unreachable upper bound)
  - argmin → q0.50 →  33.03 | vs median  +0.00 [+0.00, +0.00] | vs mean  -1.07 [-5.69, +1.86]
  - bias0  → q0.50 →  33.03 | vs median  +0.00 [+0.00, +0.00] | vs mean  -1.07 [-5.69, +1.86]
  - cov50  → q0.50 →  33.03 | vs median  +0.00 [+0.00, +0.00] | vs mean  -1.07 [-5.69, +1.86]

**Contamination check**: 3 of 244 calibration traces are outside `TRAIN_IDXS` (the finetuning val set). argmin on those only: q0.50 (23.10); argmin on all 244: q0.50 (22.03).

### mmr_euclid k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50 **argmin** **bias0** **cov50** | 27.50 | -1.70 | 0.51 | 18.77 | 29.64 |
| q0.60  | 29.98 | +8.25 | 0.66 | 16.68 | 31.89 |
| q0.70  | 36.45 | +19.07 | 0.75 | 22.81 | 39.65 |
| q0.80  | 47.56 | +32.76 | 0.84 | 35.23 | 52.10 |
| q0.90  | 68.82 | +55.59 | 0.93 | 59.95 | 73.53 |
| q0.95  | 97.06 | +83.94 | 0.98 | 92.55 | 100.59 |
| q0.99  | 215.48 | +198.22 | 1.00 | 225.96 | 213.34 |
| mean _test-argmin_ | 29.71 | +7.52 | 0.63 | 15.63 | 32.34 |

**5-fold selection** (stratified by tail level) picks: q0.50, q0.50, q0.50, q0.50, q0.50 → unanimous
Held-out calibration RMSE: median 27.50 | mean 29.71 | CV-selected q 27.50

**Transfer ID** (baselines: median 18.77, mean 15.63; test-argmin mean → 15.63 = unreachable upper bound)
  - argmin → q0.50 →  18.77 | vs median  +0.00 [+0.00, +0.00] | vs mean  +3.13 [-2.89, +6.46]
  - bias0  → q0.50 →  18.77 | vs median  +0.00 [+0.00, +0.00] | vs mean  +3.13 [-2.89, +6.46]
  - cov50  → q0.50 →  18.77 | vs median  +0.00 [+0.00, +0.00] | vs mean  +3.13 [-2.89, +6.46]
**Transfer OOD** (baselines: median 29.64, mean 32.34; test-argmin mean → 32.34 = unreachable upper bound)
  - argmin → q0.50 →  29.64 | vs median  +0.00 [+0.00, +0.00] | vs mean  -2.69 [-11.31, +4.60]
  - bias0  → q0.50 →  29.64 | vs median  +0.00 [+0.00, +0.00] | vs mean  -2.69 [-11.31, +4.60]
  - cov50  → q0.50 →  29.64 | vs median  +0.00 [+0.00, +0.00] | vs mean  -2.69 [-11.31, +4.60]

**Contamination check**: 3 of 244 calibration traces are outside `TRAIN_IDXS` (the finetuning val set). argmin on those only: mean (11.38); argmin on all 244: q0.50 (27.50).

### ctx_level k=5

| spec | calib RMSE (n=244) | calib bias | calib P(pred>true) | bench ID (n=6) | bench OOD (n=5) |
|---|---|---|---|---|---|
| q0.50 **argmin** **bias0** **cov50** | 21.23 | -0.76 | 0.56 | 36.24 | 26.84 |
| q0.60  | 23.67 | +8.81 | 0.70 | 29.05 | 25.15 |
| q0.70  | 30.17 | +18.84 | 0.81 | 23.31 | 29.17 |
| q0.80 _test-argmin_ | 41.30 | +31.50 | 0.88 | 20.92 | 39.53 |
| q0.90  | 62.80 | +52.83 | 0.95 | 30.05 | 60.31 |
| q0.95  | 91.05 | +79.36 | 0.98 | 51.01 | 88.09 |
| q0.99  | 205.21 | +184.18 | 1.00 | 154.25 | 206.19 |
| mean  | 22.75 | +7.47 | 0.67 | 29.29 | 24.58 |

**5-fold selection** (stratified by tail level) picks: q0.50, q0.50, q0.50, q0.50, q0.50 → unanimous
Held-out calibration RMSE: median 21.23 | mean 22.75 | CV-selected q 21.23

**Transfer ID** (baselines: median 36.24, mean 29.29; test-argmin q0.80 → 20.92 = unreachable upper bound)
  - argmin → q0.50 →  36.24 | vs median  +0.00 [+0.00, +0.00] | vs mean  +6.95 [+4.57, +9.14]
  - bias0  → q0.50 →  36.24 | vs median  +0.00 [+0.00, +0.00] | vs mean  +6.95 [+4.57, +9.14]
  - cov50  → q0.50 →  36.24 | vs median  +0.00 [+0.00, +0.00] | vs mean  +6.95 [+4.57, +9.14]
**Transfer OOD** (baselines: median 26.84, mean 24.58; test-argmin q0.80 → 39.53 = unreachable upper bound)
  - argmin → q0.50 →  26.84 | vs median  +0.00 [+0.00, +0.00] | vs mean  +2.26 [-9.72, +8.36]
  - bias0  → q0.50 →  26.84 | vs median  +0.00 [+0.00, +0.00] | vs mean  +2.26 [-9.72, +8.36]
  - cov50  → q0.50 →  26.84 | vs median  +0.00 [+0.00, +0.00] | vs mean  +2.26 [-9.72, +8.36]

**Contamination check**: 3 of 244 calibration traces are outside `TRAIN_IDXS` (the finetuning val set). argmin on those only: q0.50 (12.74); argmin on all 244: q0.50 (21.23).


## Level-conditional optimum

Calibration tail levels (n=244): p10 34.8 | p25 64.1 | median 89.7 | p75 127.5 | p90 148.6

- benchmark ID levels ['70.4', '74.9', '113.5', '142.0', '145.9', '146.2'] → pool percentiles ['29%', '31%', '64%', '85%', '87%', '87%']
- benchmark OOD levels ['66.0', '72.2', '101.2', '156.2', '184.3'] → pool percentiles ['26%', '30%', '57%', '92%', '98%']


Argmin quantile per calibration level quartile — plus the ceiling a *level-aware* picker would reach if it knew which quartile the query is in (`oracle-by-level`), against the best single global quantile.

| model | config | Q1 | Q2 | Q3 | Q4 | best global | global RMSE | oracle-by-level RMSE | gap |
|---|---|---|---|---|---|---|---|---|---|
| chronos2 | zeroshot k=0 | q0.80 | mean | mean | q0.70 | mean | 58.18 | 57.61 | +0.57 |
| chronos2 | mmr_euclid k=5 | q0.50 | q0.50 | q0.60 | q0.70 | q0.60 | 25.93 | 21.10 | +4.84 |
| chronos2 | ctx_level k=5 | q0.50 | q0.50 | mean | q0.70 | q0.60 | 23.32 | 19.99 | +3.33 |
| chronos_bolt | zeroshot k=0 | q0.70 | q0.60 | q0.60 | q0.70 | q0.60 | 61.72 | 60.78 | +0.93 |
| chronos_bolt | mmr_euclid k=5 | q0.50 | q0.50 | mean | q0.60 | q0.50 | 28.99 | 25.94 | +3.05 |
| chronos_bolt | ctx_level k=5 | q0.50 | q0.50 | q0.50 | q0.60 | q0.50 | 24.00 | 23.49 | +0.51 |
| tirex | zeroshot k=0 | q0.70 | q0.70 | q0.70 | q0.70 | q0.70 | 31.26 | 31.26 | -0.00 |
| tirex | mmr_euclid k=5 | mean | q0.50 | mean | q0.70 | mean | 25.91 | 23.18 | +2.73 |
| tirex | ctx_level k=5 | q0.50 | q0.50 | mean | q0.70 | q0.60 | 21.33 | 18.82 | +2.51 |
| timesfm | zeroshot k=0 | q0.80 | q0.80 | q0.80 | q0.80 | q0.80 | 38.13 | 38.13 | +0.00 |
| timesfm | mmr_euclid k=5 | q0.50 | meanhead | q0.60 | q0.70 | q0.60 | 29.13 | 23.20 | +5.93 |
| timesfm | ctx_level k=5 | meanhead | meanhead | q0.60 | q0.70 | q0.60 | 21.68 | 19.06 | +2.61 |
| ft_chronos2 | zeroshot k=0 | q0.50 | q0.50 | q0.50 | q0.50 | q0.50 | 22.03 | 22.03 | +0.00 |
| ft_chronos2 | mmr_euclid k=5 | q0.50 | q0.50 | q0.50 | q0.70 | q0.50 | 27.50 | 25.56 | +1.94 |
| ft_chronos2 | ctx_level k=5 | q0.50 | q0.50 | q0.50 | mean | q0.50 | 21.23 | 19.88 | +1.35 |


## Summary — calibration-selected decoding vs. the shipped knob

| model | config | argmin | bias0 | cov50 | fold picks | q*_test | ID median | ID mean | ID @argmin | ID @bias0 | ID @q*_test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chronos2 | zeroshot k=0 | mean | q0.80 | q0.80 | mean | q0.80 | 109.91 | 89.51 | **89.51** | 67.23 | _67.23_ |
| chronos2 | mmr_euclid k=5 | q0.60 | q0.60 | q0.60 | q0.60 | q0.60 | 29.40 | 27.06 | **21.28** | 21.28 | _21.28_ |
| chronos2 | ctx_level k=5 | q0.60 | q0.60 | q0.60 | q0.60 | q0.80 | 43.14 | 40.56 | **37.11** | 37.11 | _23.23_ |
| chronos_bolt | zeroshot k=0 | q0.60 | q0.80 | q0.80 | q0.60/q0.70 | q0.80 | 111.65 | 109.03 | **103.88** | 87.48 | _87.48_ |
| chronos_bolt | mmr_euclid k=5 | q0.50 | mean | mean | mean/q0.50 | q0.60 | 26.35 | 25.86 | **26.35** | 25.86 | _21.74_ |
| chronos_bolt | ctx_level k=5 | q0.50 | q0.50 | q0.50 | q0.50 | q0.70 | 39.29 | 38.69 | **39.29** | 39.29 | _27.72_ |
| tirex | zeroshot k=0 | q0.70 | q0.70 | q0.70 | q0.70 | q0.80 | 78.92 | 75.67 | **45.70** | 45.70 | _32.77_ |
| tirex | mmr_euclid k=5 | mean | q0.60 | q0.50 | mean | q0.70 | 35.51 | 36.50 | **36.50** | 25.25 | _20.63_ |
| tirex | ctx_level k=5 | q0.60 | q0.60 | q0.60 | q0.60 | q0.80 | 40.38 | 39.33 | **34.08** | 34.08 | _21.63_ |
| timesfm | zeroshot k=0 | q0.80 | q0.80 | q0.90 | q0.80 | q0.90 | 97.54 | 95.88 | **66.86** | 66.86 | _44.44_ |
| timesfm | mmr_euclid k=5 | q0.60 | q0.60 | mean | q0.50/q0.60 | q0.60 | 28.46 | 28.46 | **21.03** | 21.03 | _21.03_ |
| timesfm | ctx_level k=5 | q0.60 | q0.60 | q0.60 | q0.60 | q0.80 | 42.53 | 42.41 | **34.63** | 34.63 | _26.26_ |
| ft_chronos2 | zeroshot k=0 | q0.50 | q0.50 | q0.50 | q0.50 | q0.70 | 25.33 | 22.20 | **25.33** | 25.33 | _16.00_ |
| ft_chronos2 | mmr_euclid k=5 | q0.50 | q0.50 | q0.50 | q0.50 | mean | 18.77 | 15.63 | **18.77** | 18.77 | _15.63_ |
| ft_chronos2 | ctx_level k=5 | q0.50 | q0.50 | q0.50 | q0.50 | q0.80 | 36.24 | 29.29 | **36.24** | 36.24 | _20.92_ |

## Mechanism: does the optimal level track the residual under-prediction?

The 06-14 claim is that "decode higher" is a global level-bias correction whose size tracks how badly the config under-predicts. On the calibration set that is directly measurable: median-decoding bias vs. the selected quantile.

| model | config | median-decode calib bias | calib RMSE @median | q*_cal |
|---|---|---|---|---|
| chronos2 | zeroshot k=0 | -61.14 | 77.42 | mean |
| chronos2 | mmr_euclid k=5 | -13.61 | 29.35 | q0.60 |
| chronos2 | ctx_level k=5 | -10.51 | 25.07 | q0.60 |
| chronos_bolt | zeroshot k=0 | -56.92 | 68.44 | q0.60 |
| chronos_bolt | mmr_euclid k=5 | -4.47 | 28.99 | q0.50 |
| chronos_bolt | ctx_level k=5 | +1.99 | 24.00 | q0.50 |
| tirex | zeroshot k=0 | -41.10 | 50.39 | q0.70 |
| tirex | mmr_euclid k=5 | -6.57 | 26.13 | mean |
| tirex | ctx_level k=5 | -7.17 | 22.52 | q0.60 |
| timesfm | zeroshot k=0 | -59.63 | 68.26 | q0.80 |
| timesfm | mmr_euclid k=5 | -8.73 | 29.58 | q0.60 |
| timesfm | ctx_level k=5 | -9.88 | 23.72 | q0.60 |
| ft_chronos2 | zeroshot k=0 | +7.05 | 22.03 | q0.50 |
| ft_chronos2 | mmr_euclid k=5 | -1.70 | 27.50 | q0.50 |
| ft_chronos2 | ctx_level k=5 | -0.76 | 21.23 | q0.50 |

Pearson r(median-decode bias, q*_cal) = -0.816 over 13 (model, config) cells — negative = more under-prediction wants a higher quantile.
