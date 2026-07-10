from typing import Any

import pandas as pd


class FeatureEngine:
    def compute_rolling_windows(self, df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
        ...

    def technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def volatility_estimates(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def error_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def market_condition_dummies(self, df: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
        ...

    def build_pipeline(self, df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        ...
