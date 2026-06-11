from pathlib import Path

import ast

from matplotlib import pyplot as plt

__all__ = ["plot_loss_curve"]


def plot_loss_curve(predictor_path: str, save_path: Path | None = None) -> None:
    """Plot loss curve from finetuning log.

    Args:
        predictor_path (str): The path of the model store path.
        save_path (Path): The path to save the loss curve plot.
    """
    log_data: list[dict] = extract_loss_log_data(Path(predictor_path))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    losses = [d["loss"] for d in log_data if "loss" in d]
    lrs = [d["learning_rate"] for d in log_data if "learning_rate" in d]

    ax1.plot(losses, color="blue", label="Train Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_xticks(
        ticks=range(0, len(losses) + 1, len(losses) // 10),
        labels=range(0, len(losses) * 100 + 1, len(losses) * 10),  # type: ignore
    )
    ax1.set_ylabel("Train Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(lrs, color="red", label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    fig.suptitle("Loss Curve and Learning Rate")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path / "loss_curve.png")
    plt.show()


def extract_loss_log_data(predictor_path: Path) -> list[dict]:
    """Extract loss logs from finetuning log.

    Args:
        predictor_path (Path): The path of the model store path

    Returns:
        list[dict]: The lost of loss log data.
    """
    possible_log_file_paths: list[Path] = list(
        Path(predictor_path).glob("**/predictor_log.txt")
    )
    if not possible_log_file_paths:
        return []

    log_file_path: Path = possible_log_file_paths[0]
    log_data: str = log_file_path.read_text()
    log_lines: list[str] = log_data.splitlines()

    log_dicts: list[dict] = [
        ast.literal_eval(line)
        for line in log_lines
        if line.startswith("{") and line.endswith("}")
    ]
    return log_dicts


if __name__ == "__main__":
    predictor_path = "c:\\Users\\sever\\code\\academics\\master\\fusiontimeseries\\src\\fusiontimeseries\\finetuning\\AutogluonModels\\ag-"
    save_path = r"C:\Users\sever\code\academics\master\fusiontimeseries\results\plots\20260101_194727_Chronos2_T2_finetuning"

    plot_loss_curve(predictor_path, Path(save_path.replace("\\", "/")))
