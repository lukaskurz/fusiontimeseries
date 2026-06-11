import numpy as np
from scipy import stats

__all__ = ["rmse_with_standard_error"]


def rmse_with_standard_error(y_true, y_pred) -> tuple[float, float]:
    """
    Compute RMSE with standard error using Delta method approximation.

    Parameters
    ----------
    y_true : array-like
        Ground truth values (averaged last 80 targets)
    y_pred : array-like
        Predicted values (averaged last 80 predictions)

    Returns
    -------
    rmse : float
        Root Mean Squared Error
    se_rmse : float
        Standard error of RMSE via Delta method
    """
    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    # Standard error of the mean squared error
    se_mse = stats.sem(squared_errors)

    # Delta method approximation for SE of RMSE
    # SE(√MSE) ≈ SE(MSE) / (2√MSE)
    se_rmse = se_mse / (2 * rmse)

    return rmse, se_rmse
