"""Operating-parameter plumbing for the few-shot example pool (Phase 0).

The four operating parameters (q, shat, rlt, rln) live only in the big
``data/flux_data.json`` dump under the ``gyroswin_{train,val,id,ood}`` splits
(255 entries), keyed by re-coded ids (1000+, 2000+, 3000+, 4000+) that are
PERMUTED with respect to the raw ``fluxes_{i}.dat`` iteration ids. This module
value-matches every raw trace (301 files) and every benchmark trace (6 ID +
5 OOD) against the dump entries and persists the result as a small tracked
JSON (``operating_params_mapping.json``, next to this module) with the
parameters embedded — so runtime lookups never need the raw data, the dump,
or any environment variable.

Matching criteria (verified during planning):
- raw <-> dump: full 800-step comparison, normalized maxdiff
  ``max|a-b| / max(1, max|a|) < 1e-6`` (actual matches are exact, 0.0).
- benchmark <-> dump: 266-step benchmark trace vs ``energy_flux[2::3]``,
  normalized maxdiff < 1e-3 (ID traces sit at ~1.6e-5 float32 rounding,
  OOD traces are exact).
- Uniqueness: second-best candidate must be > 1e-2 away.

Coverage is recorded, not asserted: the dump has 255 entries while only 251
raw traces pass the validity filter, and the 5 OOD benchmark traces have no
raw ``.dat`` counterpart — some pool examples therefore legitimately end up
with ``operating_params=None`` (Phase-2 op-kNN must filter those).
"""

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ID_TEST_RAW_IDS",
    "MAPPING_PATH",
    "OP_NAMES",
    "build_mapping",
    "load_mapping",
    "get_params_for_raw_id",
    "get_params_for_pool_index",
    "get_params_for_benchmark_trace",
    "normalize_params",
    "raw_id_for_pool_index",
    "pool_index_for_raw_id",
]

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_RAW_DIR: Path = REPO_ROOT / "data" / "flux" / "raw"
DEFAULT_DUMP_PATH: Path = REPO_ROOT / "data" / "flux_data.json"
DEFAULT_BENCHMARK_PATH: Path = REPO_ROOT / "data" / "flux" / "benchmark" / "flux_data.json"
MAPPING_PATH: Path = Path(__file__).resolve().parent / "operating_params_mapping.json"

OP_NAMES: tuple[str, ...] = ("q", "shat", "rlt", "rln")
GYROSWIN_SPLITS: tuple[str, ...] = (
    "gyroswin_train",
    "gyroswin_val",
    "gyroswin_id",
    "gyroswin_ood",
)

#: Raw iteration ids of the six in-distribution benchmark traces. These are the
#: ids that must be excluded from the few-shot example pool to avoid leakage.
ID_TEST_RAW_IDS: frozenset[int] = frozenset({8, 115, 131, 148, 235, 262})

RAW_MATCH_TOLERANCE: float = 1e-6
BENCHMARK_MATCH_TOLERANCE: float = 1e-3
SECOND_BEST_MIN_DISTANCE: float = 1e-2

#: Validity filter constants — must mirror
#: ``fusiontimeseries.finetuning.preprocessing.utils.get_valid_flux_traces``.
VALIDITY_HORIZON: int = 240
SUBSAMPLE_FACTOR: int = 3


def _normalized_maxdiff(a: NDArray, b: NDArray) -> float:
    """Scale-normalized maximum absolute difference between two series."""
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(a)))))


def _match_against_dump(
    trace: NDArray,
    dump_series: dict[tuple[str, str], NDArray],
    tolerance: float,
) -> tuple[str, str, float] | None:
    """Find the unique dump entry matching ``trace`` by value.

    Args:
        trace: Series to match.
        dump_series: ``(split, key) -> series`` candidates (same length as trace).
        tolerance: Normalized-maxdiff threshold for a match.

    Returns:
        ``(split, key, maxdiff)`` of the unique match, or None if no candidate
        is within tolerance.

    Raises:
        AssertionError: If the second-best candidate is suspiciously close
            (ambiguous match).
    """
    diffs = sorted(
        ((_normalized_maxdiff(trace, series), split, key) for (split, key), series in dump_series.items()),
    )
    best_diff, best_split, best_key = diffs[0]
    if best_diff >= tolerance:
        return None
    second_diff = diffs[1][0]
    assert second_diff > SECOND_BEST_MIN_DISTANCE, (
        f"Ambiguous match for {best_split}/{best_key}: "
        f"best={best_diff:.2e}, second-best={second_diff:.2e}"
    )
    return best_split, best_key, best_diff


def build_mapping(
    raw_dir: Path | None = None,
    dump_path: Path | None = None,
    benchmark_path: Path | None = None,
    out_path: Path = MAPPING_PATH,
) -> dict:
    """Value-match raw and benchmark traces against the gyroswin dump entries.

    Writes the mapping JSON to ``out_path`` and returns it. Requires the raw
    ``.dat`` files, the big dump, and the benchmark JSON on disk (build time
    only — runtime lookups use the persisted JSON).

    Args:
        raw_dir: Directory with ``fluxes_{i}.dat`` files.
        dump_path: Path to the big ``data/flux_data.json`` dump.
        benchmark_path: Path to our benchmark ``flux_data.json``.
        out_path: Where to write the mapping JSON.

    Returns:
        The mapping dict (same content as the written JSON).
    """
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    dump_path = dump_path or DEFAULT_DUMP_PATH
    benchmark_path = benchmark_path or DEFAULT_BENCHMARK_PATH

    dump: dict = json.load(open(dump_path, "r"))
    dump_series: dict[tuple[str, str], NDArray] = {}
    dump_params: dict[tuple[str, str], dict[str, float]] = {}
    for split in GYROSWIN_SPLITS:
        for key, entry in dump[split].items():
            dump_series[(split, key)] = np.asarray(entry["energy_flux"], dtype=np.float64)
            dump_params[(split, key)] = {name: float(entry[name]) for name in OP_NAMES}
    dump_series_subsampled: dict[tuple[str, str], NDArray] = {
        sk: series[2::SUBSAMPLE_FACTOR] for sk, series in dump_series.items()
    }

    ########################################################
    # Raw traces: full-length exact matching + pool indices
    ########################################################
    n_raw = len(list(raw_dir.glob("fluxes_*.dat")))
    raw_traces: dict[str, dict] = {}
    matched_dump_keys: set[tuple[str, str]] = set()
    pool_index = 0
    for raw_id in range(n_raw):
        flux = np.loadtxt(raw_dir / f"fluxes_{raw_id}.dat")[:, 1]

        # Same validity filter as get_valid_flux_traces()
        is_valid = (
            float(np.mean(flux[:VALIDITY_HORIZON])) >= 1.0
            and float(np.mean(flux[-VALIDITY_HORIZON:])) >= 1.0
        )

        match = _match_against_dump(flux, dump_series, RAW_MATCH_TOLERANCE)
        if match is not None:
            split, key, maxdiff = match
            assert (split, key) not in matched_dump_keys, (
                f"Dump entry {split}/{key} matched by more than one raw trace "
                f"(second: raw {raw_id})"
            )
            matched_dump_keys.add((split, key))
            raw_traces[str(raw_id)] = {
                "pool_index": pool_index if is_valid else None,
                "dump_split": split,
                "dump_key": key,
                "params": dump_params[(split, key)],
                "match_maxdiff": maxdiff,
            }
        else:
            raw_traces[str(raw_id)] = {
                "pool_index": pool_index if is_valid else None,
                "dump_split": None,
                "dump_key": None,
                "params": None,
                "match_maxdiff": None,
            }
        if is_valid:
            pool_index += 1
    n_valid = pool_index

    ########################################################
    # Benchmark traces: [2::3]-subsampled matching
    ########################################################
    benchmark: dict = json.load(open(benchmark_path, "r"))
    benchmark_traces: dict[str, dict] = {}
    benchmark_splits: dict[str, str] = {}
    for json_key, kind in (("in_distribution", "id"), ("out_of_distribution", "ood")):
        for trace_key, values in benchmark[json_key].items():
            trace = np.asarray(values, dtype=np.float64)
            match = _match_against_dump(trace, dump_series_subsampled, BENCHMARK_MATCH_TOLERANCE)
            assert match is not None, f"Benchmark trace {trace_key} has no dump match"
            split, key, maxdiff = match
            raw_id = int(trace_key.split("_")[1 if kind == "id" else 2]) if kind == "id" else None
            benchmark_traces[trace_key] = {
                "kind": kind,
                "raw_id": raw_id,
                "dump_split": split,
                "dump_key": key,
                "params": dump_params[(split, key)],
                "match_maxdiff": maxdiff,
            }
            benchmark_splits[trace_key] = split

    # ID <-> gyroswin_id and OOD <-> gyroswin_ood must be bijective per split
    for kind, expected_split in (("id", "gyroswin_id"), ("ood", "gyroswin_ood")):
        entries = {k: v for k, v in benchmark_traces.items() if v["kind"] == kind}
        keys = {v["dump_key"] for v in entries.values()}
        assert all(v["dump_split"] == expected_split for v in entries.values()), (
            f"Benchmark {kind} traces matched outside {expected_split}"
        )
        assert len(keys) == len(entries) == len(dump[expected_split]), (
            f"Benchmark {kind} <-> {expected_split} not bijective: "
            f"{len(keys)} unique keys for {len(entries)} traces, "
            f"{len(dump[expected_split])} dump entries"
        )

    # Consistency: benchmark iteration_N and raw trace N resolve to the same dump key
    for trace_key, entry in benchmark_traces.items():
        if entry["raw_id"] is None:
            continue
        raw_entry = raw_traces[str(entry["raw_id"])]
        assert raw_entry["dump_key"] == entry["dump_key"] and raw_entry["dump_split"] == entry["dump_split"], (
            f"Inconsistent match: benchmark {trace_key} -> "
            f"{entry['dump_split']}/{entry['dump_key']} but raw {entry['raw_id']} -> "
            f"{raw_entry['dump_split']}/{raw_entry['dump_key']}"
        )

    ID_test_pool_coverage = sum(
        1 for raw_id in ID_TEST_RAW_IDS if raw_traces[str(raw_id)]["pool_index"] is not None
    )
    mapping = {
        "meta": {
            "n_raw_traces": n_raw,
            "n_valid_raw_traces": n_valid,
            "n_dump_entries": len(dump_series),
            "n_raw_matched": len(matched_dump_keys),
            "n_valid_matched": sum(
                1
                for v in raw_traces.values()
                if v["pool_index"] is not None and v["dump_key"] is not None
            ),
            "n_benchmark_traces": len(benchmark_traces),
            "n_id_test_traces_valid": ID_test_pool_coverage,
            "criterion": {
                "metric": "max|a-b| / max(1, max|a|)",
                "raw_tolerance": RAW_MATCH_TOLERANCE,
                "benchmark_tolerance": BENCHMARK_MATCH_TOLERANCE,
                "second_best_min_distance": SECOND_BEST_MIN_DISTANCE,
                "benchmark_subsampling": "energy_flux[2::3]",
            },
        },
        "raw_traces": raw_traces,
        "benchmark_traces": benchmark_traces,
    }

    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=1)
    return mapping


########################################################
# Runtime lookup API (uses only the persisted JSON)
########################################################


@cache
def load_mapping() -> dict:
    """Load the persisted mapping JSON (cached).

    Raises:
        FileNotFoundError: If the mapping has not been built yet.
    """
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Operating-params mapping not found at {MAPPING_PATH}. "
            "Build it with: uv run python -m "
            "fusiontimeseries.benchmarking.few_shot.operating_params"
        )
    return json.load(open(MAPPING_PATH, "r"))


@cache
def _pool_index_to_raw_id() -> dict[int, int]:
    """Reverse map pool_index -> raw iteration id."""
    return {
        entry["pool_index"]: int(raw_id)
        for raw_id, entry in load_mapping()["raw_traces"].items()
        if entry["pool_index"] is not None
    }


def get_params_for_raw_id(raw_id: int) -> dict[str, float] | None:
    """Operating parameters for a raw iteration id, or None if unmatched.

    Raises:
        KeyError: If ``raw_id`` is not a known raw iteration id.
    """
    return load_mapping()["raw_traces"][str(raw_id)]["params"]


def get_params_for_pool_index(pool_index: int) -> dict[str, float] | None:
    """Operating parameters for a pool index (get_valid_flux_traces key)."""
    return get_params_for_raw_id(raw_id_for_pool_index(pool_index))


def get_params_for_benchmark_trace(trace_key: str) -> dict[str, float]:
    """Operating parameters for a benchmark trace key (all 11 are matched).

    Raises:
        KeyError: If ``trace_key`` is not a known benchmark trace.
    """
    return load_mapping()["benchmark_traces"][trace_key]["params"]


def raw_id_for_pool_index(pool_index: int) -> int:
    """Raw iteration id for a pool index.

    Raises:
        KeyError: If ``pool_index`` is not a valid pool index.
    """
    return _pool_index_to_raw_id()[pool_index]


def pool_index_for_raw_id(raw_id: int) -> int | None:
    """Pool index for a raw iteration id, or None if the trace is invalid.

    Raises:
        KeyError: If ``raw_id`` is not a known raw iteration id.
    """
    return load_mapping()["raw_traces"][str(raw_id)]["pool_index"]


def normalize_params(params: dict[str, float]) -> dict[str, float]:
    """Min-max normalize operating parameters to [0, 1] using FTSConfig ranges."""
    from fusiontimeseries.lib.config import FTSConfig

    op_ranges = FTSConfig().op_ranges
    return {
        name: (params[name] - lo) / (hi - lo)
        for name, (lo, hi) in op_ranges.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/inspect the operating-params mapping")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if the mapping exists")
    args = parser.parse_args()

    if args.rebuild or not MAPPING_PATH.exists():
        print(f"Building mapping -> {MAPPING_PATH}")
        build_mapping()
        load_mapping.cache_clear()
        _pool_index_to_raw_id.cache_clear()
    else:
        print(f"Mapping exists at {MAPPING_PATH} (use --rebuild to rebuild)")

    mapping = load_mapping()
    meta = mapping["meta"]
    print("\n--- Coverage ---")
    for key, value in meta.items():
        if key != "criterion":
            print(f"  {key}: {value}")

    print("\n--- Smoke test ---")
    it8 = mapping["benchmark_traces"]["iteration_8_ifft"]
    assert it8["dump_split"] == "gyroswin_id" and it8["dump_key"] == "3003", (
        f"Expected iteration_8_ifft -> gyroswin_id/3003, got {it8['dump_split']}/{it8['dump_key']}"
    )
    print(f"  iteration_8_ifft -> {it8['dump_split']}/{it8['dump_key']} ✓")

    params_train = get_params_for_pool_index(0)
    params_id = get_params_for_benchmark_trace("iteration_8_ifft")
    params_ood = get_params_for_benchmark_trace("ood_iteration_0_ifft_realpotens")
    print(f"  pool_index 0 (raw {raw_id_for_pool_index(0)}): {params_train}")
    print(f"  iteration_8_ifft: {params_id}")
    print(f"  ood_iteration_0_ifft_realpotens: {params_ood}")

    for label, params in (("id", params_id), ("ood", params_ood)):
        if params is None:
            continue
        normalized = normalize_params(params)
        assert all(-0.05 <= v <= 1.05 for v in normalized.values()), (
            f"Normalized {label} params out of range: {normalized}"
        )
        print(f"  normalized {label}: " + ", ".join(f"{k}={v:.3f}" for k, v in normalized.items()))

    excluded_pool_indices = sorted(
        pool_index_for_raw_id(raw_id)
        for raw_id in ID_TEST_RAW_IDS
        if pool_index_for_raw_id(raw_id) is not None
    )
    print(f"  ID test raw ids {sorted(ID_TEST_RAW_IDS)} -> pool indices {excluded_pool_indices}")

    print("\n✅ Operating-params mapping OK")
