"""In-context finetuning (ICF) dataset for Chronos-2 BilinearLoRA (Phase 9).

Phase 6 showed the finetuned model's in-context ability is INHERITED from base
pretraining — `train_bilinear.py` finetunes on single subsampled traces, never
on demonstrations. This dataset trains the BilinearLoRA on multi-example ICL
*concatenations* so the model learns to USE demonstrations, paired with the
same level-clean retrieval used at inference.

Each training sample is built from one query trace plus ``k`` demonstration
traces drawn from the OTHER traces in the split:

    [demo_1 (full), demo_2 (full), ..., demo_k (full), query_context] -> future

mirroring the eval-time flat-concat presentation (`make_concat_forecast_fn`,
where a pool example is its full ctx+target trace and the most-similar example
sits LAST, adjacent to the query). The concatenation is RAW (no scaling):
Chronos-2 instance-norms the whole stream internally and independently of a
uniform outer scaler (the Phase-3/4 finding), so raw concatenation at train is
the level-clean equivalent of ``make_concat_forecast_fn(..., "shared")`` at
eval — both preserve the demos' levels relative to the query, both get
instance-normed as one stream. Per-example scaling (the Phase-1 default) would
erase exactly the level signal ICF is meant to teach.

Two retrieval modes (the two ICF checkpoints):

- ``"level"``: the ``k`` demos nearest the query in absolute context LEVEL
  (``|mean(demo[:start]) - mean(query[:start])|``), so the train-time
  demonstration distribution matches inference-time ``ctx_level`` retrieval.
- ``"random"``: ``k`` demos sampled at random — the control that probes whether
  the model learns to USE level-matched demos (level) or merely tolerates
  demonstrations regardless of level (random).

Conditioning is QUERY-ONLY (Option A): one ``operating_parameters (B, 4)``
vector per forward = the query's params (`ConditionedTrainer.compute_loss`
patches a single vector; there is no per-segment conditioning). ``k`` is sampled
per-sample from a small set (default {1, 3, 5}) so the model is robust to the
k=5/10 used at eval.

Validation samples are DETERMINISTIC (per-idx local RNG, no data augmentation)
so ``eval_loss`` is a stable model-selection signal; training samples use the
global RNG (advancing) with white-noise augmentation on the query window, like
the base recipe. The local-RNG-in-val design never reseeds the global RNG, so
periodic evaluation does not contaminate the training sample sequence.

Self-tests (CPU, synthetic traces, no model):
    uv run python -m fusiontimeseries.finetuning.chronos2.icl_dataset
"""

import numpy as np
import torch

from fusiontimeseries.finetuning.chronos2.dataset import Chronos2Dataset

__all__ = ["Chronos2ICLDataset"]


def _zscore_context(x: np.ndarray) -> np.ndarray:
    """Per-sample z-score, an exact copy of ``selection._zscore``.

    Duplicated rather than imported to keep the finetuning package free of a
    dependency on the benchmarking package; the ``shape`` retrieval key must
    match the eval-side ``ctx_euclid`` key exactly.
    """
    mean, std = float(x.mean()), float(x.std())
    std = std if std > 0 else 1.0
    return (x - mean) / std


class Chronos2ICLDataset(Chronos2Dataset):
    """Chronos-2 dataset that builds multi-example ICL concatenations.

    Class-level defaults are overridden per run by ``train_val_split`` (so the
    train and val instances share the same retrieval mode / k-set).

    Attributes:
        icl_retrieval: ``"level"`` (nearest by context level), ``"shape"``
            (nearest by Euclidean distance between context windows) or
            ``"random"``.
        exclude_siblings: drop demo candidates from the SAME simulation as the
            query. The flat flux data expands every simulation into its three
            subsample phases, which share operating parameters AND saturation
            level, so a sibling demo hands the model the query's own answer.
            Without this, ``shape`` retrieval returns a sibling 83% of the time
            (median |tail gap| 0.04 against a pool spread of 42) and the model
            learns to copy the last demo's level instead of inferring it.
        num_examples: k-set sampled per training sample (e.g. ``(1, 3, 5)``).
        val_num_examples: fixed k used for the deterministic validation samples.
        icl_start_context: window for the context-level retrieval key (80, the
            eval ``start_context_length`` — the early-context mean is the
            strongest level predictor, Phase-7 ρ ≈ +0.89).
    """

    icl_retrieval: str = "level"
    exclude_siblings: bool = True
    num_examples: tuple[int, ...] = (1, 3, 5)
    val_num_examples: int = 3
    icl_start_context: int = 80

    def __init__(self, time_series, operating_parameters, config, mode) -> None:
        super().__init__(time_series, operating_parameters, config, mode)
        # Per-trace context level = mean of the first ``icl_start_context``
        # steps (the eval ``ctx_level`` retrieval key). Cheap; precompute once.
        self.context_levels: np.ndarray = np.array(
            [
                float(np.mean(np.asarray(ts)[: self.icl_start_context]))
                for ts in self.time_series
            ],
            dtype=np.float64,
        )
        # Per-trace Z-SCORED context window = the eval ``ctx_euclid`` retrieval
        # key. The z-scoring is what separates SHAPE from LEVEL: on raw
        # contexts the level offset dominates the Euclidean distance and
        # ``shape`` retrieval degenerates into ``level`` retrieval. Matches
        # ``benchmarking.few_shot.selection._zscore`` exactly.
        # Per-trace simulation id, taken from the operating-parameter vector
        # (exactly 3 traces per distinct vector = the 3 subsample phases).
        self.sim_ids: np.ndarray = np.unique(
            np.asarray(self.ops, dtype=np.float64), axis=0, return_inverse=True
        )[1]
        self.context_windows: np.ndarray = np.stack(
            [
                _zscore_context(np.asarray(ts, dtype=np.float64)[: self.icl_start_context])
                for ts in self.time_series
            ]
        )

    @classmethod
    def train_val_split(
        cls,
        config,
        *,
        icl_retrieval: str = "level",
        num_examples: tuple[int, ...] = (1, 3, 5),
        val_num_examples: int = 3,
        exclude_siblings: bool = True,
    ):
        """Base split, then inject the ICF retrieval mode + k-set on both sets.

        Args:
            config: ``FTSConfig`` (with ``context_length`` raised to the ICF
                window, e.g. 2048).
            icl_retrieval: ``"level"``, ``"shape"`` or ``"random"`` (one per
                ICF checkpoint).
            num_examples: k-set sampled per training sample.
            val_num_examples: fixed k for deterministic validation samples.

        Returns:
            ``(train_dataset, val_dataset)``, both ``Chronos2ICLDataset``.
        """
        if icl_retrieval not in ("level", "shape", "random"):
            raise ValueError(
                f"Unknown icl_retrieval {icl_retrieval!r}; "
                "expected 'level', 'shape' or 'random'"
            )
        train_dataset, val_dataset = super().train_val_split(config)
        for dataset in (train_dataset, val_dataset):
            dataset.icl_retrieval = icl_retrieval
            dataset.exclude_siblings = bool(exclude_siblings)
            dataset.num_examples = tuple(num_examples)
            dataset.val_num_examples = int(val_num_examples)
        return train_dataset, val_dataset

    def _select_demos(
        self, idx: int, k: int, query: np.ndarray, rng
    ) -> list[int]:
        """Pick ``k`` demonstration trace indices (!= idx), most-similar LAST.

        ``level`` ranks the other traces by ``|context_level - query_level|``
        and returns them nearest-LAST (the nearest demo sits adjacent to the
        query, matching the ``selection.py`` convention). ``shape`` ranks them
        by the Euclidean distance between Z-SCORED 80-step context windows,
        the train-side counterpart of the eval ``ctx_euclid`` retriever, and
        orders them the same way. ``random`` samples without replacement via ``rng``
        (the global module in train mode, a deterministic per-idx ``Generator``
        in val mode).
        """
        candidates = np.array(
            [
                j
                for j in range(len(self.time_series))
                if j != idx
                and not (self.exclude_siblings and self.sim_ids[j] == self.sim_ids[idx])
            ]
        )
        if self.icl_retrieval == "level":
            query_level = float(np.mean(np.asarray(query)[: self.icl_start_context]))
            dists = np.abs(self.context_levels[candidates] - query_level)
            order = np.argsort(dists, kind="stable")  # nearest first
            chosen = candidates[order[:k]]
            return [int(j) for j in chosen[::-1]]  # nearest LAST
        if self.icl_retrieval == "shape":
            query_ctx = _zscore_context(
                np.asarray(query, dtype=np.float64)[: self.icl_start_context]
            )
            dists = np.linalg.norm(
                self.context_windows[candidates] - query_ctx, axis=1
            )
            order = np.argsort(dists, kind="stable")  # nearest first
            chosen = candidates[order[:k]]
            return [int(j) for j in chosen[::-1]]  # nearest LAST
        if self.icl_retrieval == "random":
            chosen = rng.choice(candidates, size=k, replace=False)
            return [int(j) for j in np.atleast_1d(chosen)]
        raise ValueError(f"Unknown icl_retrieval {self.icl_retrieval!r}")

    def _build_icl_sample(
        self, idx: int
    ) -> tuple[np.ndarray, list[int], int, np.ndarray, np.ndarray]:
        """Build the raw ICL concatenation and the query future target.

        Returns:
            ``(concat, demo_indices, cutoff, history, aug_ops)`` where ``concat``
            is the raw ``[demos..., query_context]`` stream (float32), ``cutoff``
            is the query context length, ``history`` is the (possibly augmented)
            query context, and ``aug_ops`` is the (possibly augmented) query
            operating-parameter vector. Train mode uses the advancing global
            RNG + white-noise augmentation; val mode uses a deterministic
            per-idx ``Generator`` and NO augmentation.
        """
        query = np.asarray(self.time_series[idx])
        query_ops = self.ops[idx, ...]
        length = len(query)
        pred_len = self.config.prediction_length

        if self.mode == "train":
            rng = np.random
            cutoff = int(np.random.randint(pred_len, length - pred_len + 1))
            history, aug_ops = self.apply_data_augmentation(query[:cutoff], query_ops)
            k = int(np.random.choice(self.num_examples))
        else:  # val: deterministic per-idx, no augmentation
            rng = np.random.default_rng([int(self.config.random_seed), int(idx)])
            cutoff = int(rng.integers(pred_len, length - pred_len + 1))
            history, aug_ops = query[:cutoff].copy(), query_ops.copy()
            k = int(self.val_num_examples)

        demo_indices = self._select_demos(idx, k, query, rng)
        segments = [
            np.asarray(self.time_series[j], dtype=np.float64) for j in demo_indices
        ]
        segments.append(np.asarray(history, dtype=np.float64))
        concat = np.concatenate(segments).astype(np.float32)
        return concat, demo_indices, cutoff, np.asarray(history), np.asarray(aug_ops)

    def prepare_sample(self, idx: int) -> dict[str, torch.Tensor]:
        concat, _demo_indices, cutoff, _history, aug_ops = self._build_icl_sample(idx)
        context_length = self.config.context_length

        context = torch.full(
            size=(context_length,), fill_value=self.config.padding_value
        )  # NaN
        context_mask = torch.full_like(
            context, fill_value=self.config.padding_mask_default
        )  # 0.0
        # Left-truncate if the concat overflows the window (matches
        # pipeline.predict's clamp); NaN-left-pad otherwise.
        used = concat[-context_length:]
        context[-len(used):] = torch.tensor(used)
        context_mask[-len(used):] = self.config.padding_mask_indicator  # 1.0

        query = np.asarray(self.time_series[idx])
        target = query[cutoff : cutoff + self.config.prediction_length]
        future = torch.Tensor(np.asarray(target, dtype=np.float64))
        future_mask = torch.ones_like(future)

        return {
            "context": context,
            "context_mask": context_mask,
            "future_target": future,
            "future_target_mask": future_mask,
            "operating_parameters": torch.Tensor(aug_ops),
        }


########################################################
# CPU self-tests (synthetic traces, no model)
########################################################

if __name__ == "__main__":
    from fusiontimeseries.lib.config import FTSConfig

    print("Chronos2ICLDataset self-tests (CPU, synthetic traces)...")

    CONTEXT_LENGTH = 2048
    PRED_LEN = 80
    START = 80
    N_TRACES = 24
    TRACE_LEN = 267

    rng0 = np.random.default_rng(0)
    # Synthetic traces with DISTINCT levels so level-retrieval is meaningful;
    # each trace = growth ramp into a flat saturation level + small noise.
    time_series: list[np.ndarray] = []
    levels = np.linspace(5.0, 140.0, N_TRACES)
    for level in levels:
        ramp = level * (1.0 - np.exp(-np.arange(TRACE_LEN) / 20.0))
        ts = (ramp + rng0.normal(0.0, 0.5, size=TRACE_LEN)).astype(np.float64)
        time_series.append(ts)
    # Operating params in the FTSConfig op ranges (shat, q, rlt, rln).
    ops = np.stack(
        [
            rng0.uniform(0.0, 5.0, N_TRACES),
            rng0.uniform(1.0, 9.0, N_TRACES),
            rng0.uniform(3.5, 12.0, N_TRACES),
            rng0.uniform(0.0, 7.0, N_TRACES),
        ],
        axis=1,
    )

    def make_dataset(mode: str, retrieval: str, augment: bool) -> Chronos2ICLDataset:
        config = FTSConfig(
            context_length=CONTEXT_LENGTH,
            prediction_length=PRED_LEN,
            num_ops=4,
            padding_value=torch.nan,
            padding_mask_default=0.0,
            padding_mask_indicator=1.0,
            sampling_strategy="linear",
            data_augmentation="white_noise" if augment else None,
            random_seed=123,
        )
        ds = Chronos2ICLDataset(
            time_series=list(time_series),
            operating_parameters=ops.copy(),
            config=config,
            mode=mode,
        )
        ds.icl_retrieval = retrieval
        ds.num_examples = (1, 3, 5)
        ds.val_num_examples = 3
        return ds

    # T1 — shapes + NaN-left-pad layout (train, no augmentation so we can
    # bit-check that the query context is the LAST real segment)
    ds = make_dataset("train", "level", augment=False)
    sample = ds.prepare_sample(7)
    assert sample["context"].shape == (CONTEXT_LENGTH,), sample["context"].shape
    assert sample["future_target"].shape == (PRED_LEN,), sample["future_target"].shape
    assert sample["operating_parameters"].shape == (4,)
    assert sample["context_mask"].shape == (CONTEXT_LENGTH,)
    assert sample["future_target_mask"].shape == (PRED_LEN,)
    ctx = sample["context"].numpy()
    mask = sample["context_mask"].numpy()
    n_real = int(np.isfinite(ctx).sum())
    assert int(mask.sum()) == n_real, "mask must mark exactly the finite entries"
    # real entries are a contiguous RIGHT-aligned block (NaN-left-pad)
    assert np.all(np.isnan(ctx[: CONTEXT_LENGTH - n_real])), "left pad must be NaN"
    assert np.all(np.isfinite(ctx[CONTEXT_LENGTH - n_real :])), "right block must be real"
    print(
        f"✓ T1: shapes context{tuple(sample['context'].shape)} / "
        f"future{tuple(sample['future_target'].shape)} / op(4,); "
        f"{n_real} real steps NaN-left-padded, mask matches"
    )

    # T2 — concat layout: query context is the last segment, demos precede it
    concat, demo_indices, cutoff, history, aug_ops = ds._build_icl_sample(7)
    demo_len = sum(len(ds.time_series[j]) for j in demo_indices)
    assert len(concat) == demo_len + cutoff, (
        f"concat {len(concat)} != demos {demo_len} + cutoff {cutoff}"
    )
    assert np.allclose(concat[-cutoff:], history.astype(np.float32)), (
        "query context must be the LAST segment of the concat"
    )
    expected_demos = np.concatenate(
        [np.asarray(ds.time_series[j], dtype=np.float32) for j in demo_indices]
    )
    assert np.allclose(concat[:-cutoff], expected_demos), "demos must precede the query"
    assert 7 not in demo_indices, "self-exclusion: query must not be its own demo"
    assert len(set(demo_indices)) == len(demo_indices), "demos must be distinct"
    print(
        f"✓ T2: concat = [{len(demo_indices)} demos | query ctx({cutoff})]; "
        f"query LAST, demos precede, self-excluded, distinct"
    )

    # T3 — level retrieval picks the level-nearest, most-similar LAST
    q_level = float(np.mean(time_series[7][:START]))
    others = np.array([j for j in range(N_TRACES) if j != 7])
    nearest = others[np.argmin(np.abs(ds.context_levels[others] - q_level))]
    assert demo_indices[-1] == int(nearest), (
        f"level retrieval must place the level-nearest demo LAST: "
        f"{demo_indices[-1]} vs {nearest}"
    )
    # the demo levels should be (weakly) increasing in distance toward the FIRST
    demo_dists = [abs(ds.context_levels[j] - q_level) for j in demo_indices]
    assert demo_dists == sorted(demo_dists, reverse=True), (
        f"level demos must be ordered furthest-first / nearest-last: {demo_dists}"
    )
    print(f"✓ T3: level retrieval nearest-LAST (query level {q_level:.1f})")

    # T4 — level vs random modes pick different demos
    ds_rand = make_dataset("train", "random", augment=False)
    # Fix k for a clean comparison by forcing num_examples to a single value.
    ds.num_examples = (5,)
    ds_rand.num_examples = (5,)
    np.random.seed(0)
    level_demos = ds._build_icl_sample(7)[1]
    np.random.seed(0)
    rand_demos = ds_rand._build_icl_sample(7)[1]
    assert set(level_demos) != set(rand_demos), (
        f"level and random must differ: {level_demos} vs {rand_demos}"
    )
    # level demos must be genuinely closer in level than random ones
    level_gap = np.mean([abs(ds.context_levels[j] - q_level) for j in level_demos])
    rand_gap = np.mean([abs(ds_rand.context_levels[j] - q_level) for j in rand_demos])
    assert level_gap < rand_gap, f"level demos not closer: {level_gap:.1f} vs {rand_gap:.1f}"
    print(
        f"✓ T4: level vs random differ; mean level gap {level_gap:.1f} (level) < "
        f"{rand_gap:.1f} (random)"
    )

    # T5 — validation determinism (level): repeated calls bit-identical
    ds_val = make_dataset("val", "level", augment=True)
    s1 = ds_val.prepare_sample(3)
    s2 = ds_val.prepare_sample(3)
    for key in s1:
        a, b = s1[key].numpy(), s2[key].numpy()
        assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b)), f"val non-deterministic: {key}"
    # also for random retrieval (the seeded per-idx Generator)
    ds_val_rand = make_dataset("val", "random", augment=True)
    r1 = ds_val_rand._build_icl_sample(3)[1]
    r2 = ds_val_rand._build_icl_sample(3)[1]
    assert r1 == r2, f"val random retrieval non-deterministic: {r1} vs {r2}"
    print("✓ T5: validation samples deterministic (level + random retrieval)")

    # T6 — k is sampled from the set in train mode; fixed in val
    ds_k = make_dataset("train", "level", augment=False)
    np.random.seed(1)
    ks = {len(ds_k._build_icl_sample(i % N_TRACES)[1]) for i in range(60)}
    assert ks <= {1, 3, 5} and len(ks) > 1, f"train k-set not sampled: {ks}"
    ds_valk = make_dataset("val", "level", augment=False)
    val_ks = {len(ds_valk._build_icl_sample(i)[1]) for i in range(min(N_TRACES, 10))}
    assert val_ks == {3}, f"val k must be fixed at 3: {val_ks}"
    print(f"✓ T6: train k sampled from {sorted(ks)}; val k fixed at 3")

    # T7 — future target is the query window right after the context
    ds2 = make_dataset("val", "level", augment=False)
    concat2, demos2, cutoff2, _, _ = ds2._build_icl_sample(5)
    sample2 = ds2.prepare_sample(5)
    expected_future = np.asarray(time_series[5][cutoff2 : cutoff2 + PRED_LEN], dtype=np.float32)
    assert np.allclose(sample2["future_target"].numpy(), expected_future), (
        "future_target must be query[cutoff:cutoff+pred_len]"
    )
    print("✓ T7: future_target == query[cutoff:cutoff+pred_len]")

    print("\n✅ Chronos2ICLDataset self-tests passed!")
