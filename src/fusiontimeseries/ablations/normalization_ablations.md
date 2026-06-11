


| normalization | ID RMSE +- SE     | OOD RMSE +- SE     | Best Val Loss | Best Step | Final Val Loss |
| ------------- | ----------------- | ------------------ | ------------- | --------- | -------------- |
| no-norm       | (24.5981, 6.1466) | (29.9144, 7.619)   | 33.834        | 200       | 34.474         |
| global        | (37.3811, 8.5857) | (31.8498, 14.857)  | 24.758        | 100       | 29.2069        |
| arcsinh       | (40.31, 6.8542)   | (36.3308, 16.8137) | 27.886        | 100       | 29.739         |
| base          | (28.230, 6.7611)  | (30.8099, 11.2813) | 29.995        | 100       | 32.4429        |


TimesFM normlizes by calculating mean and std on the first patch that has at least three non-padded positions in it.
Patches are bags of 32 consecutive timesteps.
due to the big distribution shift in gyrokinetic flux, calculating mean and std in the linear phase of the fusion process where flux is nearly indistinguishable from random noise centered around zero leads to massively upscaling later stages, potentially harming the training process.
completely disabling normalization and internal transformer block normalizations handle distribution shift yields better results than using any other normalization technique.
We use the resulting learning rate from the ablations with identical setup for all other hyperparameters.
the base line corresponds to the chosen learning rate ablation run.
