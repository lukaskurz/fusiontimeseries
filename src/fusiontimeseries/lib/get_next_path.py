from pathlib import Path

__all__ = ["get_next_path"]


def get_next_path(base_fname: str, base_dir: Path, separator: str = "-") -> Path:
    """Gets the next available directory path with an incremented numeric suffix.
    This function scans a directory for existing subdirectories that follow a naming
    pattern of "base_fname{separator}number" and returns a path with the next
    available number in the sequence.
    Args:
        base_fname (str): The base name for the directory (e.g., "results").
        base_dir (Path): The parent directory to search for existing directories.
        separator (str, optional): The separator between base name and number.
            Defaults to "-".
    Returns:
        Path: A Path object representing the next available directory path
            (e.g., base_dir/results-0, base_dir/results-1, etc.).
    Examples:
        >>> get_next_path("results", Path("/tmp"))
        Path('/tmp/results-0')
        >>> # If results-0, results-1 exist
        >>> get_next_path("results", Path("/tmp"))
        Path('/tmp/results-2')
    """

    """Gets the next available directory path (e.g., results-0, results-1, results-2, ...)."""
    nums = [
        int(d.name.split(separator)[-1])
        for d in base_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(f"{base_fname}{separator}")
        and d.name.split(separator)[-1].isdigit()
    ]
    return base_dir / f"{base_fname}{separator}{max(nums, default=-1) + 1}"
