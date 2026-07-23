"""
Feature engineering for the stock clustering pipeline.
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def build_returns(t_df: pd.DataFrame) -> dict:
    returns = t_df["log_return"].values
    
    ann_return = float(np.mean(returns) * 252)
    ann_vol = float(np.std(returns) * np.sqrt(252))
    sharpe = (ann_return - 0.065) / (ann_vol + 1e-6)  # 6.5% RBI repo rate baseline
    skewness = float(pd.Series(returns).skew())
    
    cum_rets = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum_rets)
    drawdown = (cum_rets - peak) / peak
    max_dd = float(np.min(drawdown))
    
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "skewness": skewness,
        "max_drawdown": max_dd,
    }

def build_market_features(t_df: pd.DataFrame) -> dict:
    returns = t_df["log_return"].values
    mkt_ret = t_df["mkt_return"].values
    
    cov = np.cov(returns, mkt_ret)
    beta = float(cov[0, 1] / (cov[1, 1] + 1e-8))
    
    return {
        "beta": beta
    }

def build_trend(t_df: pd.DataFrame) -> dict:
    close = t_df["close"].values
    
    sma20 = float(pd.Series(close).rolling(20).mean().iloc[-1])
    dist_20dma = (close[-1] - sma20) / sma20
    
    return {
        "dist_20dma": dist_20dma
    }

def build_momentum(t_df: pd.DataFrame) -> dict:
    close = t_df["close"].values
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = float((100 - (100 / (1 + rs))).dropna().iloc[-1]) if len(rs.dropna()) > 0 else 50.0
    
    return {
        "rsi_14": rsi
    }

def build_volatility(t_df: pd.DataFrame) -> dict:
    close = t_df["close"].values
    high = t_df["high"].values
    low = t_df["low"].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = float(pd.Series(tr).rolling(14).mean().dropna().iloc[-1]) if len(tr) >= 14 else 0.0
    atr_pct = atr / (close[-1] + 1e-8)
    
    return {
        "atr_pct": atr_pct
    }

def build_cluster_features(df: pd.DataFrame, cfg: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts both time-series price returns and multi-factor feature profiles per ticker.
    """
    logger.info("Extracting stock features for clustering...")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 1. Pivot price matrix for correlation distance
    pivoted_prices = df.pivot(index="date", columns="ticker", values="close")

    # 2. Derive equal-weighted market return as benchmark for Beta calculation
    df["log_return"] = df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    mkt_returns = df.groupby("date")["log_return"].mean().rename("mkt_return")
    df = df.merge(mkt_returns, on="date", how="left")

    min_ticker_rows = cfg.get("min_ticker_rows", 60) if cfg else 60

    features_list = []
    for ticker, t_df in df.groupby("ticker"):
        t_df = t_df.dropna(subset=["log_return"]).copy()
        if len(t_df) < min_ticker_rows:
            continue
            
        features = {"ticker": ticker}
        features.update(build_returns(t_df))
        features.update(build_market_features(t_df))
        features.update(build_momentum(t_df))
        features.update(build_volatility(t_df))
        features.update(build_trend(t_df))
        
        features_list.append(features)

    factor_df = pd.DataFrame(features_list).set_index("ticker").dropna()
    logger.info("Successfully extracted features for %d tickers.", len(factor_df))
    
    return pivoted_prices[factor_df.index], factor_df
