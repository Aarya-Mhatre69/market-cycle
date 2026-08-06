"""
Feature engineering for the stock clustering pipeline.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_returns_and_tail_risk(t_df: pd.DataFrame, rf_rate: float = 0.065) -> dict:
    returns = t_df["log_return"].values
    
    ann_return = float(np.mean(returns) * 252)
    ann_vol = float(np.std(returns) * np.sqrt(252)) + 1e-8
    
    sharpe = (ann_return - rf_rate) / ann_vol
    
    downside_returns = returns[returns < 0]
    downside_vol = float(np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 0 else ann_vol
    sortino = (ann_return - rf_rate) / (downside_vol + 1e-8)
    
    skewness = float(pd.Series(returns).skew())
    kurtosis = float(pd.Series(returns).kurtosis())
    var_95 = float(np.percentile(returns, 5))
    
    cum_rets = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum_rets)
    drawdown = (cum_rets - peak) / peak
    max_dd = float(np.min(drawdown))
    
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "max_drawdown": max_dd,
    }


def build_market_features(t_df: pd.DataFrame) -> dict:
    returns = t_df["log_return"].values
    mkt_ret = t_df["mkt_return"].values
    
    cov = np.cov(returns, mkt_ret)
    beta = float(cov[0, 1] / (cov[1, 1] + 1e-8))
    
    return {"beta": beta}


def build_multi_trend_and_momentum(t_df: pd.DataFrame) -> dict:
    close = t_df["close"].values
    n = len(close)
    
    sma20 = float(pd.Series(close).rolling(20).mean().iloc[-1]) if n >= 20 else close[-1]
    sma50 = float(pd.Series(close).rolling(50).mean().iloc[-1]) if n >= 50 else close[-1]
    sma200 = float(pd.Series(close).rolling(200).mean().iloc[-1]) if n >= 200 else close[-1]
    
    dist_20dma = (close[-1] - sma20) / (sma20 + 1e-8)
    dist_50dma = (close[-1] - sma50) / (sma50 + 1e-8)
    dist_200dma = (close[-1] - sma200) / (sma200 + 1e-8)
    
    mom_1m = float((close[-1] / close[-21]) - 1) if n >= 21 else 0.0
    mom_3m = float((close[-1] / close[-63]) - 1) if n >= 63 else 0.0
    
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = float((100 - (100 / (1 + rs))).dropna().iloc[-1]) if len(rs.dropna()) > 0 else 50.0
    
    return {
        "dist_20dma": dist_20dma,
        "dist_50dma": dist_50dma,
        "dist_200dma": dist_200dma,
        "mom_1m": mom_1m,
        "mom_3m": mom_3m,
        "rsi_14": rsi,
    }


def build_volatility_structure(t_df: pd.DataFrame) -> dict:
    close = t_df["close"].values
    high = t_df["high"].values
    low = t_df["low"].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = float(pd.Series(tr).rolling(14).mean().dropna().iloc[-1]) if len(tr) >= 14 else 0.0
    atr_pct = atr / (close[-1] + 1e-8)
    
    sma20 = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    upper = sma20 + (2 * std20)
    lower = sma20 - (2 * std20)
    bandwidth = float(((upper - lower) / (sma20 + 1e-8)).iloc[-1]) if len(sma20) >= 20 else 0.0
    
    return {
        "atr_pct": atr_pct,
        "bollinger_bandwidth": bandwidth,
    }


def build_liquidity_and_volume_features(t_df: pd.DataFrame) -> dict:
    close = t_df["close"].values
    volume = t_df["volume"].values
    returns = np.abs(t_df["log_return"].values)
    
    turnover = close * volume
    vol_20d = float(pd.Series(volume).rolling(20).mean().iloc[-1]) if len(volume) >= 20 else volume[-1]
    vol_50d = float(pd.Series(volume).rolling(50).mean().iloc[-1]) if len(volume) >= 50 else volume[-1]
    volume_surge_ratio = vol_20d / (vol_50d + 1e-8)
    
    amihud = float(np.mean(returns / (turnover + 1e-5)) * 1e6)
    
    turnover_cv = float(np.std(turnover) / (np.mean(turnover) + 1e-8))
    
    return {
        "volume_surge_ratio": volume_surge_ratio,
        "amihud_illiquidity": amihud,
        "turnover_cv": turnover_cv,
    }


def build_cluster_features(df: pd.DataFrame, cfg: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Extracting expanded multi-factor features for stock clustering...")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    pivoted_prices = df.pivot(index="date", columns="ticker", values="close")

    df["log_return"] = df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    mkt_returns = df.groupby("date")["log_return"].mean().rename("mkt_return")
    df = df.merge(mkt_returns, on="date", how="left")

    min_ticker_rows = cfg.get("min_ticker_rows", 60) if cfg else 60
    rf_rate = cfg.get("risk_free_rate", 0.065) if cfg else 0.065

    features_list = []
    for ticker, t_df in df.groupby("ticker"):
        t_df = t_df.dropna(subset=["log_return"]).copy()
        if len(t_df) < min_ticker_rows:
            continue
        
        features = {"ticker": ticker}
        features.update(build_returns_and_tail_risk(t_df, rf_rate=rf_rate))
        features.update(build_market_features(t_df))
        features.update(build_multi_trend_and_momentum(t_df))
        features.update(build_volatility_structure(t_df))
        features.update(build_liquidity_and_volume_features(t_df))
        
        features_list.append(features)

    factor_df = pd.DataFrame(features_list).set_index("ticker").dropna()
    logger.info("Successfully extracted 20 features for %d tickers.", len(factor_df))
    
    return pivoted_prices[factor_df.index], factor_df
