import numpy as np


class PersistenceBaseline:
    def predict(self, last_value: float, horizon: int) -> np.ndarray:
        return np.full(horizon, last_value)


class HistoricalVolBaseline:
    def __init__(self, window: int = 20) -> None:
        self._window = window

    def predict(self, prices: np.ndarray, horizon: int) -> np.ndarray:
        ...
