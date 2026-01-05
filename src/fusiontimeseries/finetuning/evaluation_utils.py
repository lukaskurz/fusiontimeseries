from datetime import datetime
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from fusiontimeseries.benchmarking.benchmark_utils import rmse_with_standard_error
from fusiontimeseries.finetuning.preprocessing.utils import get_benchmark_flux_traces

__all__ = [
    "create_benchmark_dfs_from_flux_traces",
    "evaluate_forecasts",
    "plot_forecast_vs_true",
    "autoregressive_forecast",
    "FinetuningConfig",
    "FinetuningResults",
    "save_finetuning_results",
    "run_complete_evaluation",
]

TIMESERIES_CONVERSION_START_TIMESTAMP: datetime = datetime(2000, 1, 1)
TIMESERIES_CONVERSION_TIMEDELTA_UNIT: str = "ms"


class FinetuningConfig(BaseModel):
    """Configuration for finetuning experiment."""

    model_name: str
    prediction_length: int
    target: str
    eval_metric: str
    hyperparameters: dict[str, Any]
    time_limit: int | None = None
    start_context_length: int = 80
    relevant_tail_length: int = 80


class FinetuningResults(BaseModel):
    """Results from finetuning evaluation."""

    timestamp: str
    config: FinetuningConfig
    training_data_size: int = Field(description="Number of training time series")
    in_distribution: dict[str, Any] = Field(
        description="ID metrics: rmse, se_rmse, n_samples"
    )
    out_of_distribution: dict[str, Any] = Field(
        description="OOD metrics: rmse, se_rmse, n_samples"
    )
    predictor_path: str | None = Field(
        default=None, description="Path to saved predictor"
    )


def save_finetuning_results(
    results: FinetuningResults,
    output_dir: Path | None = None,
    filename_prefix: str = "finetuning",
) -> Path:
    """Save finetuning results to JSON file.

    Args:
        results (FinetuningResults): The results to save.
        output_dir (Path | None): Directory to save results. Defaults to data/ folder.
        filename_prefix (str): Prefix for the filename.

    Returns:
        Path: Path to the saved file.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = results.timestamp
    safe_model_name = (
        results.config.model_name.replace("/", "_").replace("\\", "_").replace(" ", "-")
    )
    filename = f"{timestamp}_{safe_model_name}_{filename_prefix}_results.json"
    filepath = output_dir / filename

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(
            results.model_dump(exclude_none=True),
            f,
            indent=2,
        )

    print(f"Results saved to: {filepath}")
    return filepath


def create_benchmark_dfs_from_flux_traces(
    start_timestamp: datetime = TIMESERIES_CONVERSION_START_TIMESTAMP,
    timedelta_unit: str = TIMESERIES_CONVERSION_TIMEDELTA_UNIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create benchmark DataFrames for OOD and ID data from flux traces.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - OOD benchmark DataFrame
            - ID benchmark DataFrame
    """
    benchmark_flux_traces = get_benchmark_flux_traces()

    ood_benchmark_records = []
    for item_id, flux_trace in benchmark_flux_traces["ood"].items():
        for t in range(flux_trace.shape[0]):
            ood_benchmark_records.append(
                {
                    "item_id": item_id,
                    "timestamp": pd.to_datetime(start_timestamp)
                    + pd.to_timedelta(t, unit=timedelta_unit),  # type: ignore
                    "target": flux_trace[t],
                }
            )
    ood_benchmark_flux_df = pd.DataFrame(ood_benchmark_records)

    id_benchmark_records = []
    for item_id, flux_trace in benchmark_flux_traces["id"].items():
        for t in range(flux_trace.shape[0]):
            id_benchmark_records.append(
                {
                    "item_id": item_id,
                    "timestamp": pd.to_datetime(start_timestamp)
                    + pd.to_timedelta(t, unit=timedelta_unit),  # type: ignore
                    "target": flux_trace[t],
                }
            )
    id_benchmark_flux_df = pd.DataFrame(id_benchmark_records)

    return ood_benchmark_flux_df, id_benchmark_flux_df


def evaluate_forecasts(
    benchmark_data_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    relevant_tail_length: int = 80,
) -> tuple[float, float]:
    """Evaluate forecasts using RMSE with standard error.

    Args:
        benchmark_data_df (pd.DataFrame): DataFrame containing true values with columns
            ['item_id', 'timestamp', 'target'].
        forecasts (pd.DataFrame): DataFrame containing forecasted values with columns
            ['item_id', 'timestamp', 'target'].
        relevant_tail_length (int, optional): Length of the tail to consider for evaluation.
            Defaults to 80.

    Returns:
        tuple[float, float]: RMSE and its standard error.
    """
    assert ["item_id", "timestamp", "target"] == list(benchmark_data_df.columns), (
        "DataFrame must have columns: ['item_id', 'timestamp', 'target']"
    )
    assert ["item_id", "timestamp", "target"] == list(forecasts.columns), (
        "DataFrame must have columns: ['item_id', 'timestamp', 'target']"
    )

    y_true: list[float] = []
    y_pred: list[float] = []
    for item_id in benchmark_data_df.item_id.unique():
        true_values = benchmark_data_df[benchmark_data_df["item_id"] == item_id][
            "target"
        ].values
        y_true.append(np.mean(true_values[-relevant_tail_length:]))

        forecasted_values = forecasts[forecasts["item_id"] == item_id]["target"].values
        y_pred.append(np.mean(forecasted_values[-relevant_tail_length:]))

    rmse, se_rmse = rmse_with_standard_error(np.array(y_true), np.array(y_pred))
    return rmse, se_rmse


def plot_forecast_vs_true(
    benchmark_data_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    start_context_length: int = 80,
    save_path: Path | None = None,
) -> None:
    """Plot forecast vs true values for each item_id.

    Args:
        benchmark_data_df (pd.DataFrame): The benchmark data DataFrame with columns
            ['item_id', 'timestamp', 'target'].

        forecasts (pd.DataFrame): The forecast DataFrame with columns
            ['item_id', 'timestamp', 'target'].

        start_context_length (int, optional): The length of the context history.
            Defaults to 80.
        save_path (Path | None, optional): If provided, saves the plots to this directory.
            Defaults to None.
    """
    # Create save directory if specified
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    assert ["item_id", "timestamp", "target"] == list(benchmark_data_df.columns), (
        "DataFrame must have columns: ['item_id', 'timestamp', 'target']"
    )
    assert ["item_id", "timestamp", "target"] == list(forecasts.columns), (
        "DataFrame must have columns: ['item_id', 'timestamp', 'target']"
    )

    for item_id in benchmark_data_df.item_id.unique():
        context: pd.DataFrame = benchmark_data_df[
            benchmark_data_df["item_id"] == item_id
        ].iloc[:start_context_length]

        true_future: pd.DataFrame = benchmark_data_df[
            benchmark_data_df["item_id"] == item_id
        ].iloc[start_context_length:]

        forecast: pd.DataFrame = forecasts[forecasts["item_id"] == item_id].iloc[
            start_context_length:
        ]

        plt.figure(figsize=(12, 4))

        # plot history
        plt.plot(context["timestamp"], context["target"], label="History")

        # plot prediction start
        plt.axvline(
            x=context["timestamp"].iloc[-1],
            color="gray",
            linestyle="--",
            label="Forecast Start",
        )

        # plot prediction
        plt.plot(forecast["timestamp"], forecast["target"], label="Forecast (mean)")

        # plot actual future
        plt.plot(true_future["timestamp"], true_future["target"], label="True Future")

        plt.title(f"Item ID: {item_id}")
        plt.legend()

        if save_path is not None:
            plot_file = save_path / f"forecast_vs_true_item_{item_id}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {plot_file}")
        plt.show()
        plt.close()


def autoregressive_forecast(
    benchmark_data_df: pd.DataFrame,
    predictor: TimeSeriesPredictor,
    model_name: str = "Chronos2",
    start_context_length: int = 80,
) -> pd.DataFrame:
    """Autoregressively predict future timesteps from a starting timestep.

    Args:
        benchmark_data_df (pd.DataFrame): Ground Truth dataframe with item_id, timestamp, and target columns
        predictor (TimeSeriesPredictor): prediction model

    Returns:
        pd.DataFrame: forecast dataframe with item_id, timestamp and target columns.
    """
    assert ["item_id", "timestamp", "target"] == list(benchmark_data_df.columns), (
        "DataFrame must have columns: ['item_id', 'timestamp', 'target']"
    )

    forecasts: pd.DataFrame = pd.DataFrame(columns=["item_id", "timestamp", "target"])
    # loop over all time series in the benchmark data
    for item_id in benchmark_data_df.item_id.unique():
        # identical structure as benchmark_data_df
        benchmark_trace: pd.DataFrame = benchmark_data_df[
            benchmark_data_df["item_id"] == item_id
        ]

        context: pd.DataFrame = benchmark_trace.iloc[:start_context_length]
        prediction: TimeSeriesDataFrame  # only for correct type hinting and scoping
        while context.shape[0] < benchmark_trace.shape[0]:
            prediction = predictor.predict(context, model=model_name)

            # add prediction to context
            context: pd.DataFrame = pd.concat(
                [context, prediction["mean"].to_frame(name="target").reset_index()],
                ignore_index=True,
            )

        # columns: item_id, timestamp, target
        forecast: pd.DataFrame = context.iloc[: benchmark_trace.shape[0]]
        forecasts = pd.concat([forecasts, forecast], ignore_index=True)
    return forecasts


def run_complete_evaluation(
    predictor: TimeSeriesPredictor,
    config: FinetuningConfig,
    training_data_size: int,
    predictor_path: str | None = None,
    output_dir: Path | None = None,
) -> tuple[FinetuningResults, Path, Path]:
    """Run complete evaluation and save all results.

    This function:
    1. Creates benchmark data (ID and OOD)
    2. Runs autoregressive forecasting on both
    3. Evaluates RMSE with standard error
    4. Generates and saves plots
    5. Saves results JSON file

    Args:
        predictor (TimeSeriesPredictor): The trained predictor.
        config (FinetuningConfig): Configuration used for finetuning.
        training_data_size (int): Number of training time series.
        predictor_path (str | None): Path to saved predictor.
        output_dir (Path | None): Directory to save results. Defaults to results/ folder.

    Returns:
        tuple[FinetuningResults, Path, Path]: Results object, path to JSON file, and path to plots directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up directories
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = (
        config.model_name.replace("/", "_").replace("\\", "_").replace(" ", "-")
    )
    plots_dir = output_dir / "plots" / f"{timestamp}_{safe_model_name}_finetuning"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running complete evaluation for {config.model_name}...")
    print(f"Plots will be saved to: {plots_dir}")

    # Create benchmark data
    print("Creating benchmark DataFrames...")
    ood_benchmark_df, id_benchmark_df = create_benchmark_dfs_from_flux_traces()

    # Run autoregressive forecasting for OOD
    print("Running OOD autoregressive forecasting...")
    ood_forecasts = autoregressive_forecast(
        benchmark_data_df=ood_benchmark_df,
        predictor=predictor,
        model_name=config.model_name,
        start_context_length=config.start_context_length,
    )

    # Evaluate OOD
    print("Evaluating OOD forecasts...")
    ood_rmse, ood_se_rmse = evaluate_forecasts(
        benchmark_data_df=ood_benchmark_df,
        forecasts=ood_forecasts,
        relevant_tail_length=config.relevant_tail_length,
    )

    # Plot OOD
    print("Generating OOD plots...")
    ood_plots_dir = plots_dir / "ood"
    plot_forecast_vs_true(
        benchmark_data_df=ood_benchmark_df,
        forecasts=ood_forecasts,
        start_context_length=config.start_context_length,
        save_path=ood_plots_dir,
    )

    # Run autoregressive forecasting for ID
    print("Running ID autoregressive forecasting...")
    id_forecasts = autoregressive_forecast(
        benchmark_data_df=id_benchmark_df,
        predictor=predictor,
        model_name=config.model_name,
        start_context_length=config.start_context_length,
    )

    # Evaluate ID
    print("Evaluating ID forecasts...")
    id_rmse, id_se_rmse = evaluate_forecasts(
        benchmark_data_df=id_benchmark_df,
        forecasts=id_forecasts,
        relevant_tail_length=config.relevant_tail_length,
    )

    # Plot ID
    print("Generating ID plots...")
    id_plots_dir = plots_dir / "id"
    plot_forecast_vs_true(
        benchmark_data_df=id_benchmark_df,
        forecasts=id_forecasts,
        start_context_length=config.start_context_length,
        save_path=id_plots_dir,
    )

    # Create results object
    results = FinetuningResults(
        timestamp=timestamp,
        config=config,
        training_data_size=training_data_size,
        in_distribution={
            "rmse": float(id_rmse),
            "se_rmse": float(id_se_rmse),
            "n_samples": int(id_benchmark_df.item_id.nunique()),
        },
        out_of_distribution={
            "rmse": float(ood_rmse),
            "se_rmse": float(ood_se_rmse),
            "n_samples": int(ood_benchmark_df.item_id.nunique()),
        },
        predictor_path=predictor_path,
    )

    # Save results
    print("Saving results JSON...")
    json_path = save_finetuning_results(results, output_dir=output_dir)

    print(f"\\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"{'=' * 60}")
    print(f"OOD RMSE: {ood_rmse:.4f} ± {ood_se_rmse:.4f}")
    print(f"ID RMSE:  {id_rmse:.4f} ± {id_se_rmse:.4f}")
    print(f"\\nResults saved to: {json_path}")
    print(f"Plots saved to: {plots_dir}")
    print(f"{'=' * 60}\\n")

    return results, json_path, plots_dir
