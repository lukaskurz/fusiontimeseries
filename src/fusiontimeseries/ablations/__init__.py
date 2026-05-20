"""
We perform ablations to assess the contribution for different aspects of out method.

We use TimesFM-2.0-500M for all ablations and use the following settings for our baseline:
- 241 training samples
- 3 validation samples
- 5 ood and 6 id test samples
- One-Epoch: 241 samples, 1 sample start context 74 timesteps, stride prediction-length (64) -> one sample -> 3 samples
"""
