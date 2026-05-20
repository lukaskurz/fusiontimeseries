"""Generate markdown table summarizing experimental results."""

import json
from pathlib import Path


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """Parse folder name to extract method and context length.

    Format: METHOD-CONTEXT_LEN-DATE-RUN_NR
    """
    parts = folder_name.split("-")
    # Everything before the first numeric part is the method
    # The first numeric part is the context length

    method_parts = []
    context_len = None

    for i, part in enumerate(parts):
        if part.isdigit() and context_len is None:
            context_len = part
            break
        else:
            method_parts.append(part)

    method = "-".join(method_parts)
    return method, context_len  # type: ignore[return-value]


def read_rmse_data(results_path: Path) -> tuple[float, float]:
    """Read RMSE and standard error from results JSON file.

    Returns:
        Tuple of (rmse, standard_error)
    """
    if not results_path.exists():
        return None, None  # type: ignore[return-value]

    with open(results_path, "r") as f:
        data = json.load(f)

    # Get the first (and usually only) key in the results
    key = list(data.keys())[0]
    rmse = data[key].get("rmse")
    se = data[key].get("rmse_standard_error")

    return rmse, se


def format_table_rows(rows: list[dict]) -> list[dict]:
    """Format rows with bold highlighting for best scores within each context length group.

    Args:
        rows: List of row dictionaries with raw RMSE values

    Returns:
        List of formatted row dictionaries
    """
    from itertools import groupby

    if not rows:
        return []

    # Sort rows by context_len (as int) then by batch9_rmse_raw (descending)
    # Use negative value for descending sort, handle None values by treating them as infinity
    rows.sort(
        key=lambda r: (
            int(r["context_len"]),
            -(
                r["batch9_rmse_raw"]
                if r["batch9_rmse_raw"] is not None
                else -float("inf")
            ),
        )
    )

    # Group rows by context_len and find best scores in each group
    grouped_rows = []
    for context_len, group in groupby(rows, key=lambda r: r["context_len"]):
        group_list = list(group)

        # Find minimum RMSE for each metric in this group
        train_min = min(
            (
                r["train_rmse_raw"]
                for r in group_list
                if r["train_rmse_raw"] is not None
            ),
            default=None,
        )
        val_min = min(
            (r["val_rmse_raw"] for r in group_list if r["val_rmse_raw"] is not None),
            default=None,
        )
        id_min = min(
            (r["id_rmse_raw"] for r in group_list if r["id_rmse_raw"] is not None),
            default=None,
        )
        ood_min = min(
            (r["ood_rmse_raw"] for r in group_list if r["ood_rmse_raw"] is not None),
            default=None,
        )
        batch9_min = min(
            (
                r["batch9_rmse_raw"]
                for r in group_list
                if r["batch9_rmse_raw"] is not None
            ),
            default=None,
        )

        # Format rows with bold for best scores
        for row in group_list:
            train_is_best = (
                row["train_rmse_raw"] == train_min if train_min is not None else False
            )
            val_is_best = (
                row["val_rmse_raw"] == val_min if val_min is not None else False
            )
            id_is_best = row["id_rmse_raw"] == id_min if id_min is not None else False
            ood_is_best = (
                row["ood_rmse_raw"] == ood_min if ood_min is not None else False
            )
            batch9_is_best = (
                row["batch9_rmse_raw"] == batch9_min
                if batch9_min is not None
                else False
            )

            # Format with bold if best
            if row["train_rmse_raw"] is not None:
                train_str = f"{row['train_rmse_raw']:.2f} ± {row['train_se_raw']:.2f}"
                row["train_rmse"] = f"**{train_str}**" if train_is_best else train_str
            else:
                row["train_rmse"] = "N/A"

            if row["val_rmse_raw"] is not None:
                val_str = f"{row['val_rmse_raw']:.2f} ± {row['val_se_raw']:.2f}"
                row["val_rmse"] = f"**{val_str}**" if val_is_best else val_str
            else:
                row["val_rmse"] = "N/A"

            if row["id_rmse_raw"] is not None:
                id_str = f"{row['id_rmse_raw']:.2f} ± {row['id_se_raw']:.2f}"
                row["id_rmse"] = f"**{id_str}**" if id_is_best else id_str
            else:
                row["id_rmse"] = "N/A"

            if row["ood_rmse_raw"] is not None:
                ood_str = f"{row['ood_rmse_raw']:.2f} ± {row['ood_se_raw']:.2f}"
                row["ood_rmse"] = f"**{ood_str}**" if ood_is_best else ood_str
            else:
                row["ood_rmse"] = "N/A"

            if row["batch9_rmse_raw"] is not None:
                batch9_str = (
                    f"{row['batch9_rmse_raw']:.2f} ± {row['batch9_se_raw']:.2f}"
                )
                row["batch9_rmse"] = (
                    f"**{batch9_str}**" if batch9_is_best else batch9_str
                )
            else:
                row["batch9_rmse"] = "N/A"

            grouped_rows.append(row)

    return grouped_rows


def generate_table_markdown(rows: list[dict], title: str) -> list[str]:
    """Generate markdown table lines from formatted rows.

    Args:
        rows: List of formatted row dictionaries
        title: Title for the table section

    Returns:
        List of markdown lines
    """
    md_lines = []
    md_lines.append(f"## {title}\n")
    md_lines.append(
        "| # | Method | Context Len | Train Context Cutoffs | Subsampling | Train RMSE | Val RMSE | ID Test RMSE | OOD Test RMSE | Batch9 Test RMSE ↓ |"
    )
    md_lines.append(
        "|---|--------|-------------|----------------------|-------------|------------|----------|--------------|---------------|------------------|"
    )

    prev_context_len = None
    row_num = 1
    for row in rows:
        # Add separator line between different context lengths
        if prev_context_len is not None and row["context_len"] != prev_context_len:
            md_lines.append(
                "| --- | --------------------------- | ----------- | ---------------------------------- | ----------- | --------------   | -------------    | -------------    | ------------- | ---------------- |"
            )

        line = (
            f"| {row_num} | {row['method']} | {row['context_len']} | {row['train_context_cutoffs']} | "
            f"{row['subsampling']} | {row['train_rmse']} | {row['val_rmse']} | "
            f"{row['id_rmse']} | {row['ood_rmse']} | {row['batch9_rmse']} |"
        )
        md_lines.append(line)
        prev_context_len = row["context_len"]
        row_num += 1

    return md_lines


def generate_results_table(outputs_dir: Path, output_file: Path):
    """Generate markdown table from all run folders."""

    # Get all run folders, excluding TEST- prefixed folders
    run_folders = [
        f
        for f in outputs_dir.iterdir()
        if f.is_dir() and not f.name.startswith("TEST-")
    ]
    run_folders.sort()

    # Collect data for each run
    subsampled_rows = []
    fullcontext_rows = []
    data_scaling_rows = []

    for folder in run_folders:
        # Parse folder name
        method, context_len = parse_folder_name(folder.name)

        # Read config
        config_path = folder / "fts_config.json"
        if not config_path.exists():
            print(f"Warning: {folder.name} missing fts_config.json, skipping")
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        train_context_cutoffs = config.get("train_context_cutoffs", [])
        subsampling = config.get("subsampling", False)

        # Read results
        train_rmse, train_se = read_rmse_data(folder / "train_results.json")
        val_rmse, val_se = read_rmse_data(folder / "val_results.json")
        id_rmse, id_se = read_rmse_data(folder / "id_test_results.json")
        ood_rmse, ood_se = read_rmse_data(folder / "ood_test_results.json")
        batch9_rmse, batch9_se = read_rmse_data(folder / "batch9_test_results.json")

        # Store raw values for comparison
        row = {
            "method": method,
            "context_len": context_len,
            "train_context_cutoffs": str(train_context_cutoffs),
            "subsampling": str(subsampling),
            "train_rmse_raw": train_rmse,
            "train_se_raw": train_se,
            "val_rmse_raw": val_rmse,
            "val_se_raw": val_se,
            "id_rmse_raw": id_rmse,
            "id_se_raw": id_se,
            "ood_rmse_raw": ood_rmse,
            "ood_se_raw": ood_se,
            "batch9_rmse_raw": batch9_rmse,
            "batch9_se_raw": batch9_se,
        }

        # Separate rows into three categories
        if method.startswith("DataScaling"):
            data_scaling_rows.append(row)
        elif method.startswith("FullContext"):
            fullcontext_rows.append(row)
        else:
            subsampled_rows.append(row)

    # Format all three sets of rows
    formatted_subsampled_rows = (
        format_table_rows(subsampled_rows) if subsampled_rows else []
    )
    formatted_fullcontext_rows = (
        format_table_rows(fullcontext_rows) if fullcontext_rows else []
    )
    formatted_data_scaling_rows = (
        format_table_rows(data_scaling_rows) if data_scaling_rows else []
    )

    # Generate markdown tables
    md_lines = []
    md_lines.append("# Experimental Results Summary\n")

    # Subsampled experiments table
    if formatted_subsampled_rows:
        md_lines.extend(
            generate_table_markdown(
                formatted_subsampled_rows,
                "Subsampled Flux Timeseries / Gyroswin Setting",
            )
        )
        md_lines.append("")  # Add blank line between tables

    # Full context experiments table
    if formatted_fullcontext_rows:
        md_lines.extend(
            generate_table_markdown(formatted_fullcontext_rows, "Full Flux Timeseries")
        )
        md_lines.append("")  # Add blank line between tables

    # Data scaling experiments table
    if formatted_data_scaling_rows:
        md_lines.extend(
            generate_table_markdown(
                formatted_data_scaling_rows, "Data Scaling Experiments"
            )
        )

    # Write to file with UTF-8 encoding
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Results table written to {output_file}")
    total_rows = len(subsampled_rows) + len(fullcontext_rows) + len(data_scaling_rows)
    print(f"Processed {total_rows} run folders")
    print(f"  - Subsampled experiments: {len(subsampled_rows)}")
    print(f"  - Full context experiments: {len(fullcontext_rows)}")
    print(f"  - Data scaling experiments: {len(data_scaling_rows)}")


if __name__ == "__main__":
    # Set paths
    project_root = Path(__file__).parent.parent.parent.parent
    outputs_dir = project_root / "outputs"
    experiments_dir = Path(__file__).parent
    output_file = experiments_dir / "results_summary.md"

    generate_results_table(outputs_dir, output_file)
