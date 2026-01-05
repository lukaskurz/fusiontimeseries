# Time Series Forecasting Benchmarking Framework

To provide comparison we use the provided snapshots from non-linear simulations from GyroSwin.

We only take into account energy flux and neglect particle and momentum components. TODO Why? (because they are irrelevant at the time of experiemnts, WHY?)

Each simulation used for evaluation consists of 266 sub-sampled timesteps.
the first 80 timesteps are taken as initial context and represent the linear phase of the heat flux simulation.

To provide meaningful comparison to GyroSwin we use the same six in-distribution simulations and five out-of-distribution simulations.

We evaluate four different SOTA time series models.
We use the pretrained weights published on huggingface.
We perform the energy flux prediction autoregressively with 80 timesteps context given.
The number of timesteps predicted in one go are set to the recommended forecast window for each model.
We concatenate the predicted window with the previous context and use this as the new context window for the next prediction interval.
To evaluate the performance of autoregressive time series prediction of scalar energy flux for each model we use calculate the mean of the prediction tail set to the last 80 timesteps.
We average the predicted flux for each simulation for both in-distribution and out-of-distribution for each model and calculate the RMSE with the standard error.



#### Observations
All models tend to return the the mean of the first context window which resembles the linear phase of the energy flux.
Tirex outperforms all other models significantly on zero-shot inference.
Timeseries models collectively achieve better zero-shot performance on out-of-distribution simulations.
