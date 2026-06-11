from pathlib import Path
from typing import Any, Type

import numpy as np
import torch
from torch import nn
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

from fusiontimeseries.experiments.config import FinetuningConfig
from fusiontimeseries.lib.modules import ContinuousConditionEmbed
from fusiontimeseries.experiments.dataset import (
    FluxData,
    Namespace,
    SimulationID,
    prepare_model_input,
)
from fusiontimeseries.lib.benchmarking import rmse_with_standard_error
from fusiontimeseries.lib.conditioning import ConditionRegistry
from timesfm import TimesFmCheckpoint, TimesFmHparams
from timesfm.timesfm_torch import TimesFmTorch
from fusiontimeseries.experiments.patch import apply_patches

from fusiontimeseries.loralib.layers import (
    BilinearLoRA,
    Linear,
    RSSBilinearLoRA,
)
from fusiontimeseries.loralib.utils import mark_only_lora_as_trainable
from fusiontimeseries.loralib.utils import print_trainable_parameters


def get_model(
    config: FinetuningConfig,
    output_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    Adapter: Type[Linear | BilinearLoRA | RSSBilinearLoRA] = Linear,
) -> nn.Module:

    opc: bool = Adapter is not Linear
    if opc:
        print(
            "Using physical operating parameter conditioning with adapter:",
            Adapter.__name__,
        )

    ###################### Load pretrained TimeFM model from Hugging Face Hub ######################

    repo_id = "google/timesfm-2.0-500m-pytorch"

    hparams = TimesFmHparams(
        backend="gpu" if device == "cuda" else "cpu",
        per_core_batch_size=config.batch_size,
        horizon_len=config.prediction_length,
        context_len=config.context_length,
        num_layers=50,
        use_positional_embedding=False,
    )

    tfm = TimesFmTorch(
        hparams=hparams, checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id)
    )

    model: PatchedTimeSeriesDecoder | None = tfm._model

    if model is None:
        raise ValueError("Model is None")

    ################## Apply patches to the model ######################
    apply_patches()

    ################## Operating condition conditioning ######################
    shared_p_projection: ContinuousConditionEmbed | None = None
    if opc:
        shared_p_projection = ContinuousConditionEmbed(
            embedding_dim=512,
            n_cond=4,
            max_wavelength=10_000,
            init_weights="kaiming_uniform",
        )
        shared_p_projection.to(device)
        Adapter._shared_p_projection = shared_p_projection

    ################## Add LoRA adapters to the model ######################
    lora_model: nn.Module = Adapter.convert(
        module=model,
        kind=Adapter.__name__,
        lora_rank=8,
        lora_alpha=16,
        target_module_names=None,  # slap LoRA on all linear layers
    )

    if opc:
        # Register BEFORE mark_only_lora_as_trainable so lora_opc_embed params get detected
        lora_model.shared_condition_projection = shared_p_projection  # type: ignore

    mark_only_lora_as_trainable(model=lora_model, bias="none")

    print_trainable_parameters(
        lora_model, save_path=output_dir / "trainable_params.json"
    )

    return lora_model


def evaluate(
    model: nn.Module,
    config: FinetuningConfig,
    data: FluxData,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[Namespace, dict[str, Any]]:

    forecasts: dict[Namespace, dict[SimulationID, list[float]]] = {}
    ground_truths: dict[Namespace, dict[SimulationID, list[float]]] = {}

    for namespace, flux_data in data.items():
        forecasts[namespace] = {}
        ground_truths[namespace] = {}
        for simulation_id, flux in flux_data.items():
            energy_flux = np.array(flux["energy_flux"])

            ctx: np.ndarray = energy_flux[: config.eval_context_cutoff]
            while len(ctx) < len(energy_flux):
                with torch.no_grad():
                    sample = prepare_model_input(
                        flux=flux,
                        cutoff_idx=len(ctx),
                        config=config,
                        history_override=ctx,
                    )

                    with ConditionRegistry.patch(
                        op_params=sample["operating_parameters"].unsqueeze(0).to(device)
                    ):
                        with torch.autocast(device_type=device, dtype=torch.float32):
                            # predictions shape: (batch_size, num_patches, horizon_len, num_quantiles + 1(mean))
                            predictions: torch.Tensor = model(
                                input_ts=sample["context"]
                                .unsqueeze(0)
                                .to(device, non_blocking=True),
                                input_padding=sample["context_mask"]
                                .unsqueeze(0)
                                .to(device, non_blocking=True),
                                freq=sample["freq"]
                                .unsqueeze(0)
                                .to(device, non_blocking=True),
                            )
                forecast = (
                    predictions[0, -1, : config.prediction_length, 0].cpu().numpy()
                )
                ctx = np.concatenate([ctx, forecast])

            forecasts[namespace][simulation_id] = ctx[: len(energy_flux)].tolist()
            ground_truths[namespace][simulation_id] = energy_flux.tolist()

    results: dict[Namespace, dict[str, Any]] = {}

    prediction_tail_length: int = 80 if config.subsampling else 240
    for namespace in forecasts.keys():
        benchmark_means: list[np.floating] = [
            np.mean(gt[-prediction_tail_length:])
            for gt in ground_truths[namespace].values()
        ]
        forecast_means: list[np.floating] = [
            np.mean(fc[-prediction_tail_length:])
            for fc in forecasts[namespace].values()
        ]
        rmse, rmse_se = rmse_with_standard_error(
            y_true=np.array(benchmark_means), y_pred=np.array(forecast_means)
        )
        results[namespace] = {
            "rmse": rmse,
            "rmse_standard_error": rmse_se,
            "forecasts": forecasts[namespace],
            "ground_truths": ground_truths[namespace],
        }

    return results
