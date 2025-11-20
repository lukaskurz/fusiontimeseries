from matplotlib import pyplot as plt
import torch

__all__ = ["PredictionVizualizationProvider"]


class PredictionVizualizationProvider:
    def plot_forecast(
        self,
        ctx: torch.Tensor,
        quantile_fc: torch.Tensor,
        real_future_values: torch.Tensor | None = None,
        batch_idx: int = 0,
    ):
        """Plot a TiRex timeseries forecast.

        Args:
            ctx (torch.Tensor): Timeseries context for zero-shot forecasting.
            quantile_fc (torch.Tensor): Forecast quantiles in shape (N, forecast_len, 9)
            real_future_values (torch.Tensor | None, optional): Actual values to be forecasted. Defaults to None.
            batch_idx (int, optional): Index of the timeseries in the batch to plot. Defaults to 0.
        """
        quantile_dim: int = quantile_fc.shape[-1]
        percentages: list[float] = self.quantile_percentages(quantile_dim)

        median_forecast = quantile_fc[batch_idx, :, quantile_dim // 2].numpy()
        lower_bound = quantile_fc[batch_idx, :, 0].numpy()
        upper_bound = quantile_fc[batch_idx, :, -1].numpy()
        context = ctx[batch_idx, :].numpy()

        original_x = range(len(context))
        forecast_x = range(len(context), len(context) + len(median_forecast))

        plt.figure(figsize=(12, 6))
        plt.plot(original_x, context, label="Ground Truth Context", color="#4a90d9")
        if real_future_values is not None:
            original_fut_x = range(len(context), len(context) + len(real_future_values))
            plt.plot(
                original_fut_x,
                real_future_values,
                label="Ground Truth Future",
                color="#4a90d9",
                linestyle=":",
            )
        plt.plot(
            forecast_x,
            median_forecast,
            label=f"Forecast ({percentages[quantile_dim // 2]:.1f}% Quantile)",
            color="#d94e4e",
            linestyle="--",
        )
        plt.fill_between(
            forecast_x,
            lower_bound,
            upper_bound,
            color="#d94e4e",
            alpha=0.1,
            label=f"Forecast {percentages[0]:.1f}% - {percentages[-1]:.1f}% Quantiles",
        )
        plt.xlim(left=0)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def quantile_percentages(n):
        """
        Returns a list of quantile percentages based on the number of quantiles n.

        For example:
        n = 9  -> [10, 20, ..., 90]
        n = 10 -> [5, 15, ..., 95]
        n = 21 -> [2.5, 7.5, ..., 97.5]
        """
        step = 100 / n
        percentages = [step / 2 + step * i for i in range(n)]
        return percentages
