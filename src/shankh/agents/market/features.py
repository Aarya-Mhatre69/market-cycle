"""
Upgraded Feature engineering for the market regime pipeline.
Computes a robust 7-feature stationary matrix capturing Volatility,
Multi-period Breadth, Volume Pressure, Systemic Correlation, and Directional Momentum.
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
    logger.info("Computing upgraded market regime feature matrix...")

    cfg = cfg or {}
    sma_period = cfg.get("sma_period", 20)
    sma_long_period = cfg.get("sma_long_period", 50)
    corr_window = cfg.get("correlation_window", 20)
    min_day_data = cfg.get("min_day_data", 5)
    ann_factor = cfg.get("annualization_factor", 252)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 1. Individual Ticker Returns & Moving Averages
    df["log_return"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )

    df["sma_20"] = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(sma_period).mean()
    )
    df["sma_50"] = df.groupby("ticker")["close"].transform(
        lambda x: x.rolling(sma_long_period).mean()
    )

    df["above_sma20"] = (df["close"] > df["sma_20"]).astype(int)
    df["above_sma50"] = (df["close"] > df["sma_50"]).astype(int)
    df["is_positive"] = (df["log_return"] > 0).astype(int)

    # Dollar Volume for Up/Down Volume Ratio
    df["turnover"] = df["close"] * df["volume"]
    df["up_turnover"] = np.where(df["log_return"] > 0, df["turnover"], 0)
    df["down_turnover"] = np.where(df["log_return"] < 0, df["turnover"], 0)

    # Intraday Parkinson Volatility Component: ln(High/Low)^2 / (4 * ln(2))
    df["parkinson_var"] = (np.log(df["high"] / (df["low"] + 1e-8)) ** 2) / (4 * np.log(2))

    # Pivot return matrix for correlation density
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

        # Correlation Density across universe
        corr = window.corr().values
        triu = np.triu_indices_from(corr, k=1)
        corr_density = float(np.nanmean(corr[triu])) if len(triu[0]) > 0 else 0.0

        # Market-wide aggregations for the day
        mkt_ret_20d = float(window.mean(axis=1).sum())  # 20-day cumulative market return
        mkt_close_vol = float(window.mean(axis=1).std() * np.sqrt(ann_factor) * 100)
        
        # Parkinson Intraday Volatility (Annualized %)
        mkt_parkinson_vol = float(
            np.sqrt(day_data["parkinson_var"].mean() * ann_factor) * 100
        )

        # Composite Breadth (% of stocks above 20-DMA and 50-DMA)
        breadth_20d = float(day_data["above_sma20"].mean() * 100)
        breadth_50d = float(day_data["above_sma50"].mean() * 100)
        composite_breadth = (breadth_20d + breadth_50d) / 2.0

        # Advance-Decline Normalized Index (-1.0 to +1.0)
        advances = float(day_data["is_positive"].sum())
        declines = float(len(day_data) - advances)
        ad_index = (advances - declines) / max(advances + declines, 1.0)

        # Volume Breadth Ratio (Log ratio of Up-Volume vs Down-Volume)
        up_vol = day_data["up_turnover"].sum() + 1e-5
        down_vol = day_data["down_turnover"].sum() + 1e-5
        volume_breadth_log_ratio = float(np.log(up_vol / down_vol))

        daily_metrics.append(
            {
                "date": dt,
                "mkt_return_20d": mkt_ret_20d,
                "mkt_volatility": mkt_close_vol,
                "parkinson_volatility": mkt_parkinson_vol,
                "composite_breadth": composite_breadth,
                "breadth_pct_above_20dma": breadth_20d,
                "ad_index": ad_index,
                "volume_breadth_ratio": volume_breadth_log_ratio,
                "correlation_density": corr_density,
            }
        )

    regime_df = pd.DataFrame(daily_metrics).dropna().set_index("date")

    logger.info(
        "Generated %d daily regime observations with complete feature schema.",
        len(regime_df),
    )

    return regime_df