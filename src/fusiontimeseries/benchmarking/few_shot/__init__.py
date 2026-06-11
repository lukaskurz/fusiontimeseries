"""Few-shot in-context learning benchmarks for foundation time-series models."""

from .baselines import (
    make_knn_copy_forecast,
    make_pool_tail_mean_forecast,
    persistence_forecast,
)
from .few_shot_utils import (
    FewShotConfig,
    FewShotExample,
    create_example_pool,
    format_context_target_pairs,
    select_examples_random,
)
from .harness import (
    DEFAULT_SAVE_DIR,
    DEFAULT_SEEDS,
    FewShotRunResults,
    ForecastFn,
    PairedComparison,
    PredictFn,
    SeedResult,
    SelectFn,
    SplitSummary,
    TraceResult,
    load_results,
    make_icl_forecast_fn,
    paired_comparison,
    results_table,
    run_benchmark,
)
from .operating_params import (
    ID_TEST_RAW_IDS,
    get_params_for_benchmark_trace,
    get_params_for_pool_index,
    get_params_for_raw_id,
    load_mapping,
    normalize_params,
    pool_index_for_raw_id,
    raw_id_for_pool_index,
)

__all__ = [
    # few_shot_utils
    "FewShotConfig",
    "FewShotExample",
    "create_example_pool",
    "select_examples_random",
    "format_context_target_pairs",
    # operating_params
    "ID_TEST_RAW_IDS",
    "load_mapping",
    "get_params_for_raw_id",
    "get_params_for_pool_index",
    "get_params_for_benchmark_trace",
    "normalize_params",
    "raw_id_for_pool_index",
    "pool_index_for_raw_id",
    # harness
    "DEFAULT_SEEDS",
    "DEFAULT_SAVE_DIR",
    "PredictFn",
    "ForecastFn",
    "SelectFn",
    "make_icl_forecast_fn",
    "run_benchmark",
    "paired_comparison",
    "load_results",
    "results_table",
    "TraceResult",
    "SeedResult",
    "SplitSummary",
    "FewShotRunResults",
    "PairedComparison",
    # baselines
    "persistence_forecast",
    "make_pool_tail_mean_forecast",
    "make_knn_copy_forecast",
]
