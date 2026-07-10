import numpy as np


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def hit_rate(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual_dir = np.sign(np.diff(actual, axis=0))
    pred_dir = np.sign(np.diff(predicted, axis=0))
    return float(np.mean(actual_dir == pred_dir))
