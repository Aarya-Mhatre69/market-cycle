"""
Feature engineering for the market regime pipeline.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def build_regime_features(df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Build daily market-wide regime features from historical OHLCV data.
    """
    logger.info("Computing market-wide regime features...")

    cfg = cfg or {}
    sma_period = cfg.get("sma_period", 20)
    corr_window = cfg.get("correlation_window", 20)
    min_day_data = cfg.get("min_day_data", 5)
    ann_factor = cfg.get("annualization_factor", 252)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["log_return"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )

    df["sma"] = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(sma_period).mean()
    )

    df["above_sma"] = (df["close"] > df["sma"]).astype(int)
    df["is_positive"] = (df["log_return"] > 0).astype(int)

    pivot_returns = df.pivot(
        index="date",
        columns="ticker",
        values="log_return",
    )

    daily_metrics = []

    for i, dt in enumerate(pivot_returns.index):
        if i < corr_window:
            continue

        day_data = df[df["date"] == dt]

        if len(day_data) < min_day_data:
            continue

        window = pivot_returns.iloc[i - corr_window : i]

        corr = window.corr().values
        triu = np.triu_indices_from(corr, k=1)

        daily_metrics.append(
            {
                "date": dt,
                "mkt_return": float(day_data["log_return"].mean()),
                "mkt_volatility": float(
                    window.mean(axis=1).std() * np.sqrt(ann_factor) * 100
                ),
                "breadth_pct_above_20dma": float(
                    day_data["above_sma"].mean() * 100
                ),
                "ad_ratio": float(
                    day_data["is_positive"].sum()
                    / max(len(day_data) - day_data["is_positive"].sum(), 1)
                ),
                "correlation_density": float(np.nanmean(corr[triu])),
            }
        )

    regime_df = pd.DataFrame(daily_metrics).dropna().set_index("date")

    logger.info(
        "Generated %d daily regime observations.",
        len(regime_df),
    )

    return regime_df
