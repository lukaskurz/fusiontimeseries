"""Core utilities for few-shot in-context learning benchmarks."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from fusiontimeseries.benchmarking.zero_shot.benchmark_utils import BenchmarkConfig
from fusiontimeseries.finetuning.preprocessing.utils import get_valid_flux_traces

__all__ = [
    "FewShotConfig",
    "FewShotExample",
    "create_example_pool",
    "select_examples_random",
    "format_context_target_pairs",
]


class FewShotConfig(BenchmarkConfig):
    """Configuration for few-shot learning experiments.

    Extends BenchmarkConfig with few-shot specific parameters.
    """

    k_shot: int
    random_seed: int = 42
    example_format: Literal["context_target_pairs"] = "context_target_pairs"


class FewShotExample(BaseModel):
    """A single few-shot example trace.

    Contains the trace data split into context and target windows.
    """

    trace_id: int
    trace: list[float]  # Full trace (266 timesteps)
    context: list[float]  # First 80 timesteps
    target: list[float]  # Next 64 timesteps (timesteps 80-144)

    @property
    def trace_array(self) -> NDArray[np.float32]:
        """Return trace as numpy array."""
        return np.array(self.trace, dtype=np.float32)

    @property
    def context_array(self) -> NDArray[np.float32]:
        """Return context as numpy array."""
        return np.array(self.context, dtype=np.float32)

    @property
    def target_array(self) -> NDArray[np.float32]:
        """Return target as numpy array."""
        return np.array(self.target, dtype=np.float32)


def create_example_pool(
    exclude_ids: set[int] | None = None,
    context_length: int = 80,
    target_length: int = 64,
) -> list[FewShotExample]:
    """Create pool of candidate examples from training data.

    Uses get_valid_flux_traces() which applies:
    - Filtering: mean flux >= 1.0 at head and tail
    - Subsampling: every 3rd timestep (800 -> 266 timesteps)

    Args:
        exclude_ids: Set of trace IDs to exclude (e.g., test set IDs)
        context_length: Length of context window (default: 80)
        target_length: Length of target window (default: 64)

    Returns:
        List of FewShotExample objects
    """
    if exclude_ids is None:
        exclude_ids = set()

    # Get all valid flux traces (already subsampled)
    valid_traces: dict[int, NDArray] = get_valid_flux_traces(full_subsampling=False)

    example_pool: list[FewShotExample] = []

    for trace_id, trace in valid_traces.items():
        # Skip excluded IDs (test set)
        if trace_id in exclude_ids:
            continue

        # Extract context and target windows
        context = trace[:context_length]
        target = trace[context_length : context_length + target_length]

        # Create example
        example = FewShotExample(
            trace_id=trace_id,
            trace=trace.tolist(),
            context=context.tolist(),
            target=target.tolist(),
        )
        example_pool.append(example)

    print(
        f"Created example pool with {len(example_pool)} traces "
        f"(excluded {len(exclude_ids)} test IDs)"
    )

    return example_pool


def select_examples_random(
    pool: list[FewShotExample],
    k: int,
    seed: int = 42,
) -> list[FewShotExample]:
    """Randomly select k examples from pool.

    Args:
        pool: Pool of candidate examples
        k: Number of examples to select
        seed: Random seed for reproducibility

    Returns:
        List of k randomly selected examples
    """
    if k > len(pool):
        raise ValueError(
            f"Cannot select {k} examples from pool of size {len(pool)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pool), size=k, replace=False)
    selected = [pool[i] for i in indices]

    return selected


def format_context_target_pairs(
    examples: list[FewShotExample],
    query_context: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Format examples and query as context-target pairs for ICL.

    Format: [ex1_ctx(80), ex1_tgt(64), ex2_ctx(80), ex2_tgt(64), ..., query_ctx(80)]

    This provides full demonstration of context -> target predictions
    to the model before the query context.

    Args:
        examples: List of k example traces
        query_context: Query context array of shape [context_length]

    Returns:
        Concatenated array of shape [(k * (context_length + target_length)) + context_length]
    """
    segments: list[NDArray] = []

    # Add each example as context + target pair
    for ex in examples:
        segments.append(ex.context_array)
        segments.append(ex.target_array)

    # Add query context at the end
    segments.append(query_context)

    # Concatenate all segments
    icl_context = np.concatenate(segments)

    return icl_context


if __name__ == "__main__":
    # Example usage and testing
    print("Testing few-shot utilities...")

    # Test example pool creation
    test_ids = {8, 115, 131, 148, 235, 262}
    pool = create_example_pool(exclude_ids=test_ids)
    print(f"Pool size: {len(pool)}")

    # Verify no test IDs in pool
    pool_ids = {ex.trace_id for ex in pool}
    assert not (pool_ids & test_ids), "Test IDs found in example pool!"
    print("✓ No test set leakage")

    # Test random selection
    k = 3
    examples1 = select_examples_random(pool, k=k, seed=42)
    examples2 = select_examples_random(pool, k=k, seed=42)
    assert [e.trace_id for e in examples1] == [
        e.trace_id for e in examples2
    ], "Reproducibility failed!"
    print(f"✓ Random selection reproducible (k={k})")
    print(f"  Selected IDs: {[e.trace_id for e in examples1]}")

    # Test context-target formatting
    query_context = np.random.randn(80).astype(np.float32)
    icl_context = format_context_target_pairs(examples1, query_context)
    expected_length = k * (80 + 64) + 80
    assert (
        icl_context.shape[0] == expected_length
    ), f"ICL context length mismatch: {icl_context.shape[0]} != {expected_length}"
    print(f"✓ ICL context formatting correct (length: {icl_context.shape[0]})")

    print("\nAll tests passed!")
