from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


class ModelTrainer:
    def __init__(self, model_config: dict[str, Any]) -> None:
        self._config = model_config

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Pipeline:
        ...

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict[str, list[float]]:
        ...
