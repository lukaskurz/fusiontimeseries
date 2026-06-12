"""Finetuned Chronos-2 BilinearLoRA wrappers for the Phase-6 ICL grid.

Phase 6 (TODO numbering: "Phase 5") asks whether retrieval-ICL and
operating-param-conditioned finetuning COMPOSE: the deliverable is the 2x2
{base, finetuned} x {k=0, k-best} table. This module provides the
checkpoint-agnostic plumbing; ``run_finetuned_grid.py`` drives it and
``train_bilinear.py`` produces the (self-trained) checkpoint. Severin's
``lora_weights.pt`` can be swapped in later — every result JSON records
``checkpoint_id(path)``.

Key facts the implementation rests on (verified in-session):

- ``lora_weights.pt`` is SELF-CONTAINED: ``lora_state_dict`` keeps every key
  containing ``"lora_"``, and the shared condition embed's MLP is named
  ``lora_opc_embed`` (lib/modules.py) — so reconstruction = recipe
  architecture + ``load_state_dict(strict=False)``. The sin/cos ``omega``
  buffer is deterministic from constants.
- PARAM ORDER: conditioning tensors use the FluxData order
  **[shat, q, rlt, rln]** (``lib/dataset.py:FluxData.operating_parameters``),
  NOT the few-shot ``operating_params.OP_NAMES`` order (q, shat, rlt, rln).
  ``raw_param_tensor`` builds the vector from ``lib.config.OP_NAMES`` (the
  FluxData source of truth) so the order is correct by construction; smoke
  F2b gates it end-to-end.
- ``BilinearLoRA.forward`` RAISES when ``ConditionRegistry`` has no
  ``op_params`` and r>0 — every inference forward runs inside
  ``ConditionRegistry.patch``. The registry key is process-global, so the
  patch wraps each single ``predict`` call, not long-lived state.
- ``BilinearLoRA._shared_p_projection`` is a CLASS attribute: one finetuned
  model per process at a time (loading a second model rebinds the shared
  embed under the first one's feet).
- ``Chronos2Pipeline.predict`` silently CLAMPS the context window DOWN to
  ``model.chronos_config.context_length`` — the window knob must be set on
  the config at load time (``context_window``). The base HF config ships
  8192; training used 512.
- ``LoRALayer.convert`` drops ``lora_alpha`` in its recursion (all converted
  layers get alpha=1, scale=1/8). Training had the same behavior; loading
  uses the identical convert call so the scales match.

``severin_anchor_eval`` replicates the finetuning notebooks' evaluation
protocol EXACTLY (raw ``model(context=..., context_mask=...)`` forward,
NaN-left-padded fixed 512 window, 21-quantile median, autoregressive append
from START_IDX=80, data via ``Chronos2Dataset.get_benchmark_flux_traces``
over the rebuilt flat flux list) and scores the SAME forecasts under BOTH
metrics: the notebooks' ``mean(x[:-80])`` — which averages everything except
the tail INCLUDING the 80 copied ground-truth context steps (the README's
Chronos-2 finetuning rows, e.g. BilinearLoRA 13.83 ID, are on this easier
metric) — and the proper ``mean(x[-80:])`` tail used by our tables, the
GyroSwin paper, and Severin's own TimesFM runner.

Self-tests (CPU, fake pipeline, no model downloads):
    uv run python -m fusiontimeseries.benchmarking.few_shot.finetuned
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from fusiontimeseries.benchmarking.few_shot.covariates import (
    resolve_benchmark_trace_key,
)
from fusiontimeseries.benchmarking.few_shot.harness import ForecastFn
from fusiontimeseries.benchmarking.few_shot.operating_params import (
    get_params_for_benchmark_trace,
)
from fusiontimeseries.benchmarking.few_shot.presentation import (
    make_concat_forecast_fn,
)
from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import (
    chronos2_predict_from_pipeline,
)
from fusiontimeseries.finetuning.chronos2.train_bilinear import (
    LORA_ALPHA,
    LORA_RANK,
    TARGET_MODULE_NAMES,
    build_fts_config,
    ensure_flat_flux_data,
)
from fusiontimeseries.lib.conditioning import ConditionRegistry
from fusiontimeseries.lib.config import OP_NAMES as FLUXDATA_OP_ORDER
from fusiontimeseries.lib.modules import ContinuousConditionEmbed
from fusiontimeseries.loralib.layers import OP_PARAM_KEY, BilinearLoRA

__all__ = [
    "FINETUNED_SLUG",
    "FT_TARGET_MODULES",
    "FT_LORA_RANK",
    "FT_LORA_ALPHA",
    "FT_TRAIN_CONTEXT",
    "BASE_CONTEXT_WINDOW",
    "raw_param_tensor",
    "checkpoint_id",
    "load_finetuned_chronos2",
    "make_finetuned_forecast_fn",
    "severin_anchor_eval",
]

#: Model identity for method labels / FewShotConfig.model_slug. The variant
#: suffix stays for presentation tokens; the FINETUNED-ness lives in the slug.
FINETUNED_SLUG: str = "amazon/chronos-2-bilinear-ft"

#: Recipe constants — aliased from the training script so the reconstruction
#: convert call can never drift from the one that trained the checkpoint.
FT_TARGET_MODULES: tuple[str, ...] = tuple(TARGET_MODULE_NAMES)
FT_LORA_RANK: int = LORA_RANK
FT_LORA_ALPHA: int = LORA_ALPHA
FT_OP_EMBED_DIM: int = 512
FT_NUM_OPS: int = 4

#: Context windows: training used 512; the base HF config ships 8192 (all
#: v3-v5 base cells ran untruncated under it — a k=10 stream is 3550 steps).
FT_TRAIN_CONTEXT: int = 512
BASE_CONTEXT_WINDOW: int = 8192

#: Notebook eval protocol constant (cells 15-18).
START_IDX: int = 80


def raw_param_tensor(params: dict[str, float]) -> torch.Tensor:
    """RAW operating params as a (1, 4) conditioning tensor.

    Order is the FluxData order **[shat, q, rlt, rln]** — taken from
    ``lib.config.OP_NAMES`` (the order ``FluxData.operating_parameters``
    serializes and training consumed), NOT the few-shot
    ``operating_params.OP_NAMES`` (q, shat, rlt, rln).

    Args:
        params: Dict with keys q, shat, rlt, rln (RAW values, not normalized
            — training conditioned on raw values).

    Returns:
        Float32 tensor of shape ``(1, 4)``.
    """
    return torch.tensor(
        [[float(params[name]) for name in FLUXDATA_OP_ORDER]], dtype=torch.float32
    )


def checkpoint_id(path: Path | str) -> str:
    """``"name@sha256[:12]"`` identity of a checkpoint file."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{path.name}@{digest[:12]}"


def load_finetuned_chronos2(
    checkpoint_path: Path | str,
    device: str,
    context_window: int | None = None,
):
    """Reconstruct the BilinearLoRA-finetuned Chronos-2 from a lora_weights.pt.

    Rebuilds the exact training-time architecture (fp32 ``amazon/chronos-2``
    + ``BilinearLoRA.convert`` with the recipe constants + the shared
    ``ContinuousConditionEmbed`` wired BOTH as the class-level projection and
    as the ``shared_condition_projection`` submodule), then loads the
    self-contained LoRA state dict with hard key-set and shape asserts.

    Args:
        checkpoint_path: Path to a ``lora_weights.pt`` (``lora_state_dict``
            output of the training run).
        device: Target device.
        context_window: ``chronos_config.context_length`` override. None
            keeps the base HF value (8192, untruncated ICL streams);
            ``FT_TRAIN_CONTEXT`` (512) reproduces the training window —
            ``pipeline.predict`` silently clamps down to this value.

    Returns:
        A ``Chronos2Pipeline`` wrapping the loaded model (eval mode, fp32).
    """
    from chronos import Chronos2Pipeline
    from chronos.chronos2.model import Chronos2Model

    checkpoint_path = Path(checkpoint_path)
    model = Chronos2Model.from_pretrained("amazon/chronos-2")  # fp32
    base_window = int(model.chronos_config.context_length)
    assert base_window == BASE_CONTEXT_WINDOW, (
        f"Base chronos-2 context window changed: {base_window} != {BASE_CONTEXT_WINDOW}"
    )
    if context_window is not None:
        model.chronos_config.context_length = context_window

    shared_p_projection = ContinuousConditionEmbed(
        embedding_dim=FT_OP_EMBED_DIM,
        n_cond=FT_NUM_OPS,
        max_wavelength=10_000,
        init_weights="kaiming_uniform",
    )
    BilinearLoRA._shared_p_projection = shared_p_projection
    model = BilinearLoRA.convert(
        module=model,
        kind="BilinearLoRA",
        lora_rank=FT_LORA_RANK,
        lora_alpha=FT_LORA_ALPHA,
        target_module_names=list(FT_TARGET_MODULES),
    )
    model.shared_condition_projection = shared_p_projection

    state: dict[str, torch.Tensor] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model_lora_shapes = {
        k: tuple(v.shape) for k, v in model.state_dict().items() if "lora_" in k
    }
    ckpt_shapes = {k: tuple(v.shape) for k, v in state.items()}
    missing_in_ckpt = sorted(set(model_lora_shapes) - set(ckpt_shapes))
    unexpected_in_ckpt = sorted(set(ckpt_shapes) - set(model_lora_shapes))
    assert not missing_in_ckpt and not unexpected_in_ckpt, (
        f"Checkpoint key set mismatch: missing {missing_in_ckpt[:5]}..., "
        f"unexpected {unexpected_in_ckpt[:5]}..."
    )
    mismatched = [k for k in ckpt_shapes if ckpt_shapes[k] != model_lora_shapes[k]]
    assert not mismatched, f"Checkpoint shape mismatch: {mismatched[:5]}"

    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys[:5]
    leftover = [k for k in incompatible.missing_keys if "lora_" in k]
    assert not leftover, f"LoRA keys not loaded: {leftover[:5]}"

    model = model.to(device).eval()
    return Chronos2Pipeline(model=model)


def make_finetuned_forecast_fn(
    pipeline,
    point_stat: str = "mean",
    normalization: str = "shared",
    query_params_override: dict[str, float] | None = None,
) -> ForecastFn:
    """Harness ForecastFn for the finetuned pipeline, with OP conditioning.

    Resolves the query's RAW operating params per trace (by-value benchmark
    resolution -> ``get_params_for_benchmark_trace``), wraps every underlying
    ``predict`` call in ``ConditionRegistry.patch(op_params=...)`` (the
    BilinearLoRA forward raises without it), and otherwise delegates to the
    FROZEN Phase-3 ``make_concat_forecast_fn`` rollout — so finetuned cells
    are protocol-identical to their base twins.

    Args:
        pipeline: A ``Chronos2Pipeline`` from ``load_finetuned_chronos2``.
        point_stat: ``"mean"`` (the v5 Chronos-2 default) or ``"median"``.
        normalization: ``"shared"`` (Phase-3 winner) or ``"per_example"``.
        query_params_override: RAW params dict for non-benchmark traces
            (e.g. self-tests); benchmark traces resolve by value.

    Returns:
        A harness-compatible ForecastFn.
    """
    base_predict = chronos2_predict_from_pipeline(pipeline, point_stat)

    def forecast_fn(trace, examples, config) -> NDArray[np.float32]:
        if query_params_override is not None:
            params = dict(query_params_override)
        else:
            params = get_params_for_benchmark_trace(resolve_benchmark_trace_key(trace))
        op_params = raw_param_tensor(params)

        def conditioned_predict(
            context: NDArray[np.float32], prediction_length: int
        ) -> NDArray[np.float32]:
            with ConditionRegistry.patch(op_params=op_params):
                return base_predict(context, prediction_length)

        return make_concat_forecast_fn(conditioned_predict, normalization)(
            trace, examples, config
        )

    return forecast_fn


########################################################
# Severin-protocol anchor evaluation (notebook cells 15-18)
########################################################


def severin_anchor_eval(
    model,
    device: str,
    severin_results_path: Path | str | None = None,
) -> dict:
    """Replicate the finetuning notebooks' benchmark eval; score BOTH metrics.

    Protocol (verbatim from ``chronos2_bilinear.ipynb`` cells 15-18): for each
    of the 11 benchmark traces (``Chronos2Dataset.get_benchmark_flux_traces``,
    267-step ``[0::3]`` phase — NOT our benchmark's ``[2::3]``), start from
    the first ``START_IDX=80`` ground-truth steps, then autoregressively
    append the median (index ``n_quantiles // 2``) of a raw
    ``model(context=..., context_mask=...)`` forward over a NaN-left-padded
    fixed 512 window, conditioned on the trace's RAW params via
    ``ConditionRegistry.patch``. The notebooks score
    ``mean(x[:-80])`` — head-mean INCLUDING the 80 copied context steps —
    which this function reports as ``metrics_severin_headminus80`` alongside
    the honest ``metrics_tail80`` (``mean(x[-80:])``) on the SAME forecasts.

    Args:
        model: The loaded (finetuned) ``Chronos2Model`` — pass
            ``pipeline.model``. The raw forward ignores
            ``chronos_config.context_length``; the window is the explicit 512
            padding below.
        device: Device the model lives on.
        severin_results_path: Optional ``benchmark_results.json`` from
            Severin's run — per-trace drift is REPORTED (different run /
            device / RNG), never asserted.

    Returns:
        Dict with protocol metadata, both metric blocks, per-trace forecasts,
        and (optionally) the per-trace comparison against Severin's file.
    """
    import json

    from fusiontimeseries.finetuning.chronos2.dataset import Chronos2Dataset
    from fusiontimeseries.lib.benchmarking import rmse_with_standard_error

    fts_config = build_fts_config(device, max_steps=4000)  # notebook cell 2 values
    fts_config.data_path = ensure_flat_flux_data()
    benchmark_data = Chronos2Dataset.get_benchmark_flux_traces(fts_config)
    model = model.eval()

    forecasts: dict[str, dict[int, list[float]]] = {"id": {}, "ood": {}}
    n_quantiles: int | None = None
    for split in ("id", "ood"):
        for flux_id, flux_data in benchmark_data[split].items():
            energy_flux = np.array(flux_data.energy_flux)
            op_params = (
                torch.Tensor(flux_data.operating_parameters).unsqueeze(0).to(device)
            )
            ctx: np.ndarray = energy_flux[:START_IDX]
            while len(ctx) < len(energy_flux):
                with torch.no_grad():
                    tctx = torch.full(
                        size=(1, fts_config.context_length),
                        fill_value=fts_config.padding_value,
                    )  # NaN
                    tctx[0, -len(ctx):] = torch.tensor(ctx)
                    context_mask = torch.full_like(
                        tctx, fill_value=fts_config.padding_mask_default
                    )  # 0.0
                    context_mask[0, -len(ctx):] = fts_config.padding_mask_indicator
                    with ConditionRegistry.patch(op_params=op_params):
                        output = model(
                            context=tctx.to(device),
                            context_mask=context_mask.to(device),
                        )
                quantiles: torch.Tensor = output.quantile_preds  # (B, Qs, pred_len)
                n_quantiles = int(quantiles.shape[1])
                median_quantile: int = quantiles.shape[1] // 2
                forecast = quantiles[:, median_quantile, :].cpu().numpy().flatten()
                ctx = np.concatenate([ctx, forecast])
            forecasts[split][int(flux_id)] = ctx[: len(energy_flux)].tolist()

    def metrics(window) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for split in ("id", "ood"):
            y_true = np.array(
                [
                    float(np.mean(window(np.array(fd.energy_flux))))
                    for fd in benchmark_data[split].values()
                ]
            )
            y_pred = np.array(
                [
                    float(np.mean(window(np.array(forecasts[split][int(fid)]))))
                    for fid in benchmark_data[split].keys()
                ]
            )
            rmse, se = rmse_with_standard_error(y_true=y_true, y_pred=y_pred)
            out[split] = {"rmse": float(rmse), "standard_error": float(se)}
        return out

    result: dict = {
        "protocol": {
            "source": "chronos2_bilinear.ipynb cells 15-18 (verbatim rollout)",
            "start_idx": START_IDX,
            "window": fts_config.context_length,
            "n_quantiles": n_quantiles,
            "decoding": "median (index n_quantiles // 2)",
            "data": "Chronos2Dataset.get_benchmark_flux_traces ([0::3] phase, 267 steps)",
            "severin_metric": "mean(x[:-80]) — head mean INCLUDING the 80 copied "
            "ground-truth context steps (the notebooks' scoring bug)",
            "honest_metric": "mean(x[-80:]) — the tail mean our tables / GyroSwin / "
            "the TimesFM runner use",
        },
        "metrics_severin_headminus80": metrics(lambda x: x[:-80]),
        "metrics_tail80": metrics(lambda x: x[-80:]),
        "forecasts": forecasts,
    }

    if severin_results_path is not None:
        reference = json.load(open(severin_results_path, "r"))
        comparison: dict[str, dict[str, float]] = {}
        for split in ("id", "ood"):
            for fid, ours in forecasts[split].items():
                theirs = reference["forecasts"][split].get(str(fid))
                if theirs is None:
                    continue
                a, b = np.array(ours), np.array(theirs)
                n = min(len(a), len(b))
                rel = float(
                    np.linalg.norm(a[:n] - b[:n]) / max(1.0, np.linalg.norm(b[:n]))
                )
                comparison.setdefault(split, {})[str(fid)] = rel
        result["vs_severin"] = {
            "path": str(severin_results_path),
            "reference_metrics": reference.get("metrics"),
            "per_trace_rel_l2": comparison,
        }

    return result


########################################################
# CPU self-tests (fake pipeline, no model downloads)
########################################################

if __name__ == "__main__":
    import tempfile

    from fusiontimeseries.benchmarking.few_shot.few_shot_utils import (
        FewShotConfig,
        create_example_pool,
    )
    from fusiontimeseries.benchmarking.few_shot.harness import run_benchmark
    from fusiontimeseries.benchmarking.few_shot.operating_params import (
        ID_TEST_RAW_IDS,
    )
    from fusiontimeseries.benchmarking.few_shot.rerun_ksweep import MODEL_SLUGS
    from fusiontimeseries.benchmarking.few_shot.run_presentation_grid import (
        variant_label,
    )
    from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import (
        BenchmarkDataProvider,
    )
    from fusiontimeseries.lib.dataset import FluxData

    print("Finetuned-wrapper self-tests (CPU, fake pipeline)...")

    # T1 — param order: raw_param_tensor must equal FluxData.operating_parameters
    params = {"q": 2.0, "shat": 1.0, "rlt": 3.0, "rln": 4.0}
    flux_entry = FluxData(
        idx=0, distribution="id", shat=1.0, q=2.0, rlt=3.0, rln=4.0, energy_flux=[]
    )
    tensor = raw_param_tensor(params)
    assert tensor.shape == (1, 4) and tensor.dtype == torch.float32
    assert np.array_equal(tensor.numpy()[0], flux_entry.operating_parameters), (
        f"Param order mismatch: {tensor.numpy()[0]} vs FluxData "
        f"{flux_entry.operating_parameters}"
    )
    assert FLUXDATA_OP_ORDER == ["shat", "q", "rlt", "rln"]
    print("✓ T1: raw_param_tensor ≡ FluxData.operating_parameters order [shat, q, rlt, rln]")

    # T2 — checkpoint_id format and stability
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(b"fake checkpoint bytes")
        tmp_path = Path(f.name)
    cid1, cid2 = checkpoint_id(tmp_path), checkpoint_id(str(tmp_path))
    assert cid1 == cid2 and cid1.startswith(tmp_path.name + "@") and len(cid1.split("@")[1]) == 12
    tmp_path.unlink()
    print(f"✓ T2: checkpoint_id stable ({cid1})")

    # T3 — slug sanity: distinct from every base slug, method-label friendly
    assert FINETUNED_SLUG not in MODEL_SLUGS.values()
    assert FINETUNED_SLUG.count("/") == 1
    print(f"✓ T3: FINETUNED_SLUG {FINETUNED_SLUG!r} distinct from base slugs")

    # Fake pipeline: chronos2 layout [1, n_q, pred_len], requires the patch,
    # records every conditioning tensor it sees.
    class _FakeChronos2Pipeline:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def predict(self, inputs: torch.Tensor, prediction_length: int):
            p = ConditionRegistry.get(OP_PARAM_KEY)
            assert p is not None, "predict called outside ConditionRegistry.patch"
            self.calls.append(p.clone())
            last = float(inputs[0, 0, -1])
            base = torch.full((1, 9, prediction_length), last)
            offsets = torch.linspace(-0.5, 0.5, 9).view(1, 9, 1)
            return [base + offsets]

    fake = _FakeChronos2Pipeline()
    provider = BenchmarkDataProvider()

    def make_config(k: int) -> FewShotConfig:
        return FewShotConfig(
            device="cpu",
            model_slug=FINETUNED_SLUG,
            model_prediction_length=64,
            start_context_length=80,
            relevant_prediction_tail=80,
            k_shot=k,
            random_seed=0,
            example_target_length=None,
            normalization="shared",
            point_stat="mean",
            checkpoint="fake@000000000000",
        )

    pool = create_example_pool(exclude_ids=set(ID_TEST_RAW_IDS), target_length=None)
    examples = pool[:2]

    # T4 — per-trace param resolution + patch hygiene
    for getter, key in (
        (provider.get_id, "iteration_8_ifft"),
        (provider.get_ood, "ood_iteration_0_ifft_realpotens"),
    ):
        fake.calls.clear()
        trace = getter(key).numpy()
        fn = make_finetuned_forecast_fn(fake, point_stat="mean", normalization="shared")
        out = fn(trace, examples, make_config(2))
        expected = raw_param_tensor(get_params_for_benchmark_trace(key))
        assert out.shape == trace.shape and np.all(np.isfinite(out))
        assert len(fake.calls) >= 1
        assert all(torch.equal(c, expected) for c in fake.calls), (
            f"Conditioning tensor drifted for {key}"
        )
        assert ConditionRegistry.get(OP_PARAM_KEY) is None, "Registry not cleared"
    print("✓ T4: per-query param resolution (ID + OOD) + registry cleared after use")

    # T5 — non-benchmark traces: resolver refuses, override works
    rng = np.random.default_rng(0)
    synthetic = (5.0 + 0.1 * rng.normal(size=266)).astype(np.float32)
    fn = make_finetuned_forecast_fn(fake, point_stat="median")
    try:
        fn(synthetic, [], make_config(0))
        raise AssertionError("T5: resolver must reject a synthetic trace")
    except KeyError:
        pass
    fn_override = make_finetuned_forecast_fn(
        fake, point_stat="median", query_params_override=params
    )
    fake.calls.clear()
    out = fn_override(synthetic, [], make_config(0))
    assert out.shape == synthetic.shape
    assert all(torch.equal(c, raw_param_tensor(params)) for c in fake.calls)
    print("✓ T5: synthetic trace rejected without override, accepted with override")

    # T6 — end-to-end through the frozen harness (fake pipeline, k=1)
    results = run_benchmark(
        forecast_fn=make_finetuned_forecast_fn(fake),
        config=make_config(1),
        example_pool=pool,
        method=f"{FINETUNED_SLUG.replace('/', '_')}_random__shared-mean",
        seeds=(0,),
        provider=provider,
        save=False,
    )
    assert results.n_seeds == 1 and len(results.per_seed[0].per_trace) == 11
    assert results.config["checkpoint"] == "fake@000000000000"
    assert np.isfinite(results.in_distribution.rmse)
    print(
        f"✓ T6: fake pipeline through run_benchmark "
        f"(ID {results.in_distribution.rmse:.2f}, OOD "
        f"{results.out_of_distribution.rmse:.2f}); checkpoint recorded in config"
    )

    # T7 — label/config roundtrip incl. the win512 token
    import json as _json

    label_full = variant_label("concat", "shared", point_stat="mean")
    label_512 = variant_label(
        "concat", "shared", point_stat="mean", model_context_window=FT_TRAIN_CONTEXT
    )
    assert label_full == "shared-mean" and label_512 == "shared-mean-win512"
    config_512 = make_config(5)
    config_512.model_context_window = FT_TRAIN_CONTEXT
    roundtrip = FewShotConfig(**_json.loads(config_512.model_dump_json()))
    assert roundtrip == config_512
    print("✓ T7: variant labels (full vs win512) distinct; config JSON roundtrip ok")

    print("\n✅ Finetuned-wrapper self-tests passed!")
