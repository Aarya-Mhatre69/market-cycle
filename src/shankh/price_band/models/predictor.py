import numpy as np
from sklearn.pipeline import Pipeline


class ForecastService:
    def __init__(self, model: Pipeline, config: dict) -> None:
        self._model = model
        self._config = config

    def predict(self, features: np.ndarray) -> tuple[float, float, float]:
        ...

    def predict_with_bands(self, features: np.ndarray, confidence: float = 0.95) -> dict:
        ...
