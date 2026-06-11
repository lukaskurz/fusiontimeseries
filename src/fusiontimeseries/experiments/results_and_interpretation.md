# Experimental Results Summary

## Methods

======================================================================
Trainable Parameters by Adapter Type % (Mio.)
======================================================================

Linear              :   1.04\% (5.22 Mio.)
BilinearLoRA        :   1.25\% (6.33 Mio.)
RSSBilinearLoRA     :   1.42\% (7.17 Mio.)

======================================================================
Training Runtime by Adapter Type
======================================================================
Linear              :   565.11 ±   0.80 seconds (5 runs)
BilinearLoRA        :   725.98 ±   4.88 seconds (5 runs)
RSSBilinearLoRA     :   829.82 ±   2.15 seconds (5 runs)

======================================================================
Inference Time by Adapter Type
======================================================================
Linear              : 0.007461 ± 0.000137 seconds (75 samples)
BilinearLoRA        : 0.014379 ± 0.000261 seconds (75 samples)
RSSBilinearLoRA     : 0.016385 ± 0.000289 seconds (75 samples)

======================================================================

- **Linear**: Basic Low Rank adaption using a standard LoRA layer. This model does not have access to any operating parameters and solely uses the given context to forecast the gyrokinetic flux.
- **BilinearLoRA**: Physical Operating Parameter infused Low Rank adapter that performs pointwise interaction between embedded operating parameters and input features in low rank space.
- **RSSBilinearLoRA**: extends BilinearLoRA by performing an affine (pointwise multiplicative and additive) transfpormation in rank space.

All tables are first sorted by evaluation context length (column three) and second by Test RMSE.
We also evaluated all Gyroswin-Setting experiments on Batch 9 of the extended trajectories since Batch 9 had more samples than both ID and OOD Test sets and therefore provides a more stable identifier of the generalizability of the model.
The fact that unstable gyrokinetic flux trajectories are inherently high variance underlines this decision.
Furthermore, also evaluating the first experiments on the final testing batch allows for comparison the the Data scaling experiments.

## Subsampled Flux Timeseries / Gyroswin Setting

Table 1 shows experimental finetuning results using the Gyroswin setting. We trained on 241 flux trajectories subsampled to every third timestep (800 -> 267 timesteps).
We take each trajectory and perform different context cutoffs to get training samples. A context cutoff means that every timestep before the cutoff is given to the model as context.
The last context cutoff index was intentionally chosen to be len(trajectory) - prediction_length (128 for Timesfm).
The context len column states which context length the finetuned model was evaluated on.
For context length 1 we intentionally set the timeseries value to zero to force the model to completely depend on the operating parameters.
We clearly can see that both operating parameter conditioned models outperform the basic LoRA by far with **RSSBilinearLoRA** having the best test set RMSE.


| #   | Method                      | Context Len | Train Context Cutoffs              | Subsampling | Train RMSE       | Val RMSE         | ID Test RMSE     | OOD Test RMSE     | Test RMSE (Batch 9) ↓ |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | ---------------- | ---------------- | ---------------- | ----------------- | --------------------- |
| 1   | Linear                      | 1           | [1, 70, 139]                       | True        | 44.59 ± 1.71     | 39.40 ± 7.59     | 41.29 ± 7.22     | 53.79 ± 16.05     | 55.43 ± 3.04          |
| 2   | BilinearLoRA                | 1           | [1, 70, 139]                       | True        | **19.41 ± 1.44** | 14.99 ± 5.55     | **35.27 ± 8.07** | 33.37 ± 13.18     | 20.92 ± 5.88          |
| 3   | RSSBilinearLoRA             | 1           | [1, 70, 139]                       | True        | 19.88 ± 1.47     | **13.86 ± 3.22** | 38.44 ± 9.69     | **27.57 ± 8.96**  | **19.68 ± 5.82**      |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------    | -------------     | ----------------      |
| 4   | Linear                      | 11          | [11, 139]                          | True        | 19.93 ± 1.36     | **12.53 ± 5.30** | **34.22 ± 8.51** | 37.38 ± 14.51     | 29.75 ± 4.94          |
| 5   | BilinearLoRA                | 11          | [11, 139]                          | True        | 13.02 ± 0.98     | 20.22 ± 7.06     | 39.17 ± 10.49    | 35.54 ± 12.28     | 25.93 ± 5.70          |
| 6   | RSSBilinearLoRA             | 11          | [11, 139]                          | True        | **12.73 ± 0.83** | 14.70 ± 4.73     | 36.95 ± 12.74    | **29.93 ± 12.38** | **24.15 ± 5.41**      |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------    | -------------     | ----------------      |
| 7   | BilinearLoRA                | 21          | [21, 139]                          | True        | 157.20 ± 76.81   | **21.07 ± 2.54** | 34.55 ± 10.95    | **23.08 ± 5.48**  | 30.18 ± 5.91          |
| 8   | Linear                      | 21          | [21, 139]                          | True        | 26.77 ± 1.54     | 23.94 ± 6.99     | **30.65 ± 8.65** | 28.71 ± 10.03     | 30.04 ± 5.62          |
| 9   | RSSBilinearLoRA             | 21          | [21, 139]                          | True        | **23.69 ± 1.41** | 21.86 ± 3.89     | 33.33 ± 8.90     | 28.74 ± 6.31      | **29.52 ± 6.02**      |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------    | -------------     | ----------------      |
| 10  | Linear                      | 41          | [41, 139]                          | True        | 15.54 ± 0.94     | **21.06 ± 3.69** | **30.08 ± 9.20** | 35.89 ± 10.64     | 24.33 ± 3.77          |
| 11  | BilinearLoRA                | 41          | [41, 139]                          | True        | **11.75 ± 0.80** | 21.82 ± 6.78     | 31.34 ± 8.79     | **30.65 ± 11.36** | 24.07 ± 5.84          |
| 12  | RSSBilinearLoRA             | 41          | [41, 139]                          | True        | 12.35 ± 0.84     | 22.89 ± 3.83     | 31.25 ± 7.82     | 31.01 ± 11.94     | **23.74 ± 5.56**      |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------    | -------------     | ----------------      |
| 13  | Linear                      | 80          | [80, 139]                          | True        | 16.09 ± 0.90     | 16.42 ± 5.64     | **28.22 ± 7.93** | 35.29 ± 12.16     | 22.33 ± 5.52          |
| 14  | RSSBilinearLoRA             | 80          | [80, 139]                          | True        | 15.32 ± 0.87     | 17.19 ± 3.76     | 34.01 ± 10.61    | **31.19 ± 10.00** | 19.62 ± 4.87          |
| 15  | BilinearLoRA                | 80          | [80, 139]                          | True        | **14.53 ± 0.80** | **12.85 ± 3.16** | 35.30 ± 8.45     | 36.72 ± 10.23     | **19.03 ± 5.23**      |

## Full Flux Timeseries

The second set of experiments consists of full flux trajectories - meaning, all 800 timesteps.
Therefore we have more train context cutoffs, aka more samples to train on.
The table setup is identical to the GyroSwin setting.
Also here we clearly see the effect of physical operating parameter conditioning on the timeseries forecasting performance with reduced context size.

| #   | Method                      | Context Len | Train Context Cutoffs              | Subsampling | Train RMSE       | Val RMSE         | ID Test RMSE      | OOD Test RMSE     | Test RMSE (Batch 9) ↓ |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | ---------------- | ---------------- | ----------------- | ----------------- | --------------------- |
| 1   | FullContext-Linear          | 1           | [1, 129, 257, 385, 513, 641, 672]  | False       | 46.34 ± 1.81     | 39.62 ± 12.89    | 47.18 ± 8.80      | 58.58 ± 18.06     | 48.76 ± 2.90          |
| 2   | FullContext-RSSBilinearLoRA | 1           | [1, 129, 257, 385, 513, 641, 672]  | False       | **20.31 ± 1.44** | **13.52 ± 4.82** | 42.80 ± 11.91     | 34.05 ± 14.29     | 23.52 ± 5.41          |
| 3   | FullContext-BilinearLoRA    | 1           | [1, 129, 257, 385, 513, 641, 672]  | False       | 21.57 ± 1.48     | 16.92 ± 3.34     | **42.32 ± 12.53** | **31.41 ± 11.69** | **21.19 ± 3.89**      |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------     | -------------     | ----------------      |
| 4   | FullContext-Linear          | 11          | [11, 139, 267, 395, 523, 651, 672] | False       | 34.95 ± 1.47     | 33.38 ± 9.74     | 50.01 ± 13.71     | 48.99 ± 8.09      | 41.48 ± 4.46          |
| 5   | FullContext-RSSBilinearLoRA | 11          | [11, 139, 267, 395, 523, 651, 672] | False       | **19.96 ± 1.21** | 18.37 ± 4.52     | 38.56 ± 13.54     | **26.81 ± 8.95**  | 26.12 ± 4.45          |
| 6   | FullContext-BilinearLoRA    | 11          | [11, 139, 267, 395, 523, 651, 672] | False       | 20.02 ± 1.19     | **13.89 ± 3.35** | **38.50 ± 12.97** | 31.32 ± 11.44     | **22.06 ± 4.29**      |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------     | -------------     | ----------------      |
| 7   | FullContext-Linear          | 21          | [21, 149, 277, 405, 533, 661, 672] | False       | 24.35 ± 1.48     | **14.71 ± 4.83** | **34.19 ± 5.74**  | 28.93 ± 9.15      | 29.41 ± 5.97          |
| 8   | FullContext-RSSBilinearLoRA | 21          | [21, 149, 277, 405, 533, 661, 672] | False       | 19.53 ± 1.30     | 15.52 ± 1.44     | 37.45 ± 13.06     | 30.23 ± 14.09     | 23.79 ± 4.43          |
| 9   | FullContext-BilinearLoRA    | 21          | [21, 149, 277, 405, 533, 661, 672] | False       | **19.40 ± 1.25** | 17.41 ± 4.32     | 39.85 ± 10.26     | **24.11 ± 8.99**  | **23.09 ± 5.39**      |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------     | -------------     | ----------------      |
| 10  | FullContext-Linear          | 41          | [41, 169, 297, 425, 553, 672]      | False       | 21.53 ± 1.32     | **11.78 ± 2.85** | 42.51 ± 13.38     | 30.67 ± 9.32      | 25.79 ± 2.99          |
| 11  | FullContext-BilinearLoRA    | 41          | [41, 169, 297, 425, 553, 672]      | False       | **17.45 ± 1.05** | 15.56 ± 1.71     | **37.53 ± 14.10** | **24.77 ± 10.64** | 24.36 ± 5.61          |
| 12  | FullContext-RSSBilinearLoRA | 41          | [41, 169, 297, 425, 553, 672]      | False       | 17.52 ± 1.27     | 18.64 ± 1.62     | 40.90 ± 14.81     | 26.20 ± 10.57     | **24.21 ± 5.80**      |
| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------     | -------------     | ----------------      |
| 13  | FullContext-Linear          | 80          | [80, 208, 336, 464, 592, 672]      | False       | 19.66 ± 1.34     | 16.85 ± 3.71     | **23.25 ± 4.71**  | 33.05 ± 11.74     | 48.70 ± 15.97         |
| 14  | FullContext-BilinearLoRA    | 80          | [80, 208, 336, 464, 592, 672]      | False       | **14.76 ± 1.10** | **2.91 ± 0.99**  | 31.35 ± 5.99      | 29.97 ± 10.24     | 24.42 ± 4.75          |
| 15  | FullContext-RSSBilinearLoRA | 80          | [80, 208, 336, 464, 592, 672]      | False       | 19.25 ± 1.13     | 13.00 ± 4.67     | 34.60 ± 10.51     | **27.00 ± 11.25** | **22.04 ± 3.99**      |

## Data Scaling Experiments

For our final set of experiments we used an additional of 8 batches of gyrokinetic simulation data which provided on average 50 addtional flux trajectories per batch.
Each batch was randomly sampled and batch 6 and 9 were set for validation and testing.
For data scaling we iteratively added batches to the training data set. This is encoded in the Method: b{num} encodes the batch numbers that were included in the training dataset in this experiment run.
Our baseline forms the gyroswin training dataset of 241 flux trajectories and we interatively added batches in each run.
The most training data incorporated run 5 where all 8 additional batches were included in the training process.
We observe the best test set performance with the first 5 batches added to the training process.
The fact that more batches hurt our evaluation performance could be addressed to the fact that we did not use early stopping. The high variance in the flux trajectories kept the validation loss constantly high.

| #   | Method                                 | Context Len | Train Context Cutoffs             | Subsampling | Train RMSE       | Val RMSE (Batch 6) | ID Test RMSE      | OOD Test RMSE    | Test RMSE (Batch 9) ↓ |
| --- | -------------------------------------- | ----------- | --------------------------------- | ----------- | ---------------- | ------------------ | ----------------- | ---------------- | --------------------- |
| 1   | DataScaling-b1-RSSBilinearLoRA         | 1           | [1, 129, 257, 385, 513, 641, 672] | False       | 32.60 ± 10.18    | 21.69 ± 3.03       | 176.32 ± 85.26    | 29.72 ± 8.54     | 24.09 ± 4.36          |
| 2   | DataScaling-b12-RSSBilinearLoRA        | 1           | [1, 129, 257, 385, 513, 641, 672] | False       | 21.78 ± 1.58     | 25.03 ± 3.42       | 43.19 ± 9.29      | 37.09 ± 16.35    | 21.21 ± 5.37          |
| 3   | DataScaling-b1234-RSSBilinearLoRA      | 1           | [1, 129, 257, 385, 513, 641, 672] | False       | 19.24 ± 1.34     | 20.58 ± 3.04       | 36.45 ± 8.74      | **23.32 ± 9.53** | 20.25 ± 5.16          |
| 4   | DataScaling-b123-RSSBilinearLoRA       | 1           | [1, 129, 257, 385, 513, 641, 672] | False       | **19.22 ± 1.36** | 21.63 ± 3.10       | 40.26 ± 12.88     | 28.53 ± 13.22    | 19.55 ± 5.47          |
| 5   | DataScaling-b123457810-RSSBilinearLoRA | 1           | [1, 129, 257, 385, 513, 641, 672] | False       | 19.54 ± 1.29     | **20.25 ± 3.13**   | 39.69 ± 9.64      | 29.29 ± 10.72    | 18.76 ± 5.24          |
| 6   | DataScaling-b123457-RSSBilinearLoRA    | 1           | [1, 129, 257, 385, 513, 641, 672] | False       | 21.16 ± 1.47     | 21.31 ± 3.20       | **36.05 ± 12.02** | 35.05 ± 13.57    | 17.40 ± 4.24          |
| 7   | DataScaling-b1234578-RSSBilinearLoRA   | 1           | [1, 129, 257, 385, 513, 641, 672] | False       | 19.98 ± 1.15     | 20.68 ± 3.01       | 36.80 ± 7.92      | 25.72 ± 6.61     | 17.20 ± 3.49          |
| 8   | DataScaling-b12345-RSSBilinearLoRA     | 1           | [1, 129, 257, 385, 513, 641, 672] | False       | 20.25 ± 1.48     | 20.69 ± 3.12       | 38.29 ± 8.55      | 32.42 ± 12.22    | **16.94 ± 4.99**      |
