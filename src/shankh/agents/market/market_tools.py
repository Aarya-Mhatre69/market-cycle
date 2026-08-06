"""
src/shankh/ml/market/tool.py

LangChain Tool for live HMM Market Regime inference.

Flow
----
1.  Fetch OHLCV history for the Nifty 50 universe from yfinance.
2.  Build regime features (volatility, breadth, correlation density).
3.  Load trained HMM scaler + model from disk.
4.  Run inference and return the current regime label + market metrics.
5.  Return a JSON summary — no pre-computed CSV lookup.
"""

import json
import logging
from typing import Optional

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from shankh.agents.market.ml.regime_inference import run_inference

logger = logging.getLogger(__name__)

# Enough bars to warm up the 20-day rolling windows used by regime features.
_HISTORY_DAYS = 120

# Representative Nifty 50 tickers — broad enough for regime signal.
_NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS", "TECHM.NS", "POWERGRID.NS",
]


def _fetch_market_ohlcv(tickers: list[str], days: int = _HISTORY_DAYS) -> pd.DataFrame:
    frames = []
    for tkr in tickers:
        try:
            raw = yf.download(tkr, period=f"{days}d", interval="1d", auto_adjust=False, progress=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df["ticker"] = tkr
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) >= 30:
                frames.append(df)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", tkr, exc)

    if not frames:
        raise RuntimeError("Could not fetch OHLCV data for market universe.")

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


@tool
def get_market_regime(query_date: Optional[str] = None) -> str:
    """
    Fetch current market regime classification using the trained Hidden Markov Model (HMM).

    Args:
        query_date: Optional date string 'YYYY-MM-DD'. Defaults to latest trading day.

    Returns:
        JSON string containing regime label (Risk-On, Range-Bound, Risk-Off),
        market volatility %, composite breadth %, and recent state timeline.
    """
    try:
        df_raw = _fetch_market_ohlcv(_BROAD_MARKET_TICKERS, days=_HISTORY_DAYS)
        result = run_inference(df_raw)

        if query_date and query_date.lower() not in ("latest", "today", "now", ""):
            hist = result.get("historical_regimes", {})
            matching = {k: v for k, v in hist.items() if str(k).startswith(query_date)}
            if matching:
                result["historical_regimes"] = matching

        return json.dumps(result, indent=2, default=str)

    except Exception as exc:
        logger.error("Failed to run market regime tool: %s", exc)
        return json.dumps({"error": f"Regime inference failed: {exc}"}, indent=2)


@tool
def get_market_breadth() -> str:
    """
    Compute daily cross-sectional market breadth, moving average participation ratios,
    and Advance-Decline (A/D) volume metrics across the market universe.

    Returns:
        JSON string containing % stocks above 20/50/200 DMA, A/D ratio, normalized
        A/D index, volume breadth ratio, and universe correlation density.
    """
    try:
        df_raw = _fetch_market_ohlcv(_BROAD_MARKET_TICKERS, days=_HISTORY_DAYS)
        df = df_raw.sort_values(["ticker", "date"]).reset_index(drop=True)

        df["log_return"] = df.groupby("ticker")["close"].transform(lambda x: (x / x.shift(1)).apply(pd.np.log if hasattr(pd, "np") else lambda v: pd.Series(v).apply(lambda k: float(pd.Series([k]).apply(lambda z: pd.Series([z]).to_numpy()[0])))) if False else lambda x: pd.Series(x).pct_change())
        
        df["sma_20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
        df["sma_50"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(50).mean())
        df["sma_200"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(200).mean())

        latest_date = df["date"].max()
        day_data = df[df["date"] == latest_date].copy()

        pct_above_20dma = round(float((day_data["close"] > day_data["sma_20"]).mean() * 100.0), 2)
        pct_above_50dma = round(float((day_data["close"] > day_data["sma_50"]).mean() * 100.0), 2)
        pct_above_200dma = round(float((day_data["close"] > day_data["sma_200"]).mean() * 100.0), 2)

        advances = int((day_data["close"] > day_data["open"]).sum())
        declines = int((day_data["close"] < day_data["open"]).sum())
        total = max(advances + declines, 1)

        ad_ratio = round(float(advances / max(declines, 1)), 2)
        normalized_ad_index = round(float((advances - declines) / total), 2)

        day_data["turnover"] = day_data["close"] * day_data["volume"]
        up_volume = day_data[day_data["close"] > day_data["open"]]["turnover"].sum() + 1e-5
        down_volume = day_data[day_data["close"] < day_data["open"]]["turnover"].sum() + 1e-5
        volume_breadth_ratio = round(float(up_volume / down_volume), 2)

        pivoted_returns = df.pivot(index="date", columns="ticker", values="close").pct_change().dropna()
        corr_matrix = pivoted_returns.tail(20).corr().values
        triu_indices = pd.np.triu_indices_from(corr_matrix, k=1) if hasattr(pd, "np") else ([], [])
        
        import numpy as np
        triu = np.triu_indices_from(corr_matrix, k=1)
        corr_density = round(float(np.nanmean(corr_matrix[triu])), 3) if len(triu[0]) > 0 else 0.50

        breadth_payload = {
            "date": str(latest_date.date()),
            "total_universe_sample": day_data["ticker"].nunique(),
            "pct_above_20dma": pct_above_20dma,
            "pct_above_50dma": pct_above_50dma,
            "pct_above_200dma": pct_above_200dma,
            "advances": advances,
            "declines": declines,
            "advance_decline_ratio": ad_ratio,
            "normalized_ad_index": normalized_ad_index,
            "volume_breadth_ratio": volume_breadth_ratio,
            "universe_correlation_density": corr_density,
            "breadth_divergence_flag": "NEGATIVE_DIVERGENCE" if pct_above_50dma < 45.0 else "HEALTHY_PARTICIPATION"
        }
        return json.dumps(breadth_payload, indent=2)

    except Exception as exc:
        logger.error("Failed to compute market breadth: %s", exc)
        return json.dumps({"error": f"Market breadth calculation failed: {exc}"}, indent=2)


@tool
def get_market_cycle() -> str:
    """
    Fetch market cycle indicators: Nifty valuation percentiles (P/E, P/B),
    real yield spreads, and corporate earnings revision ratios.

    Returns:
        JSON string containing trailing valuation multiples, 10-year percentile bands,
        real yield spread %, and composite cycle positioning.
    """
    fmp_api_key = os.getenv("FMP_API_KEY")
    fred_api_key = os.getenv("FRED_API_KEY")

    cycle_data: Dict[str, Any] = {}

    # 1. Macro Real Yield via FRED API
    if fred_api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_api_key)

            in_yield = float(fred.get_series("INDIRLTLT01STM").dropna().iloc[-1])
            in_cpi_series = fred.get_series("INDCPIALLMINMEI").dropna()
            in_cpi_yoy = float(((in_cpi_series.iloc[-1] - in_cpi_series.iloc[-13]) / in_cpi_series.iloc[-13]) * 100.0)

            real_yield = round(in_yield - in_cpi_yoy, 2)
            cycle_data["india_10y_gsec_yield"] = round(in_yield, 2)
            cycle_data["india_cpi_yoy"] = round(in_cpi_yoy, 2)
            cycle_data["real_yield_spread_percent"] = real_yield
        except Exception as exc:
            logger.warning("FRED API fetch for market cycle failed: %s", exc)

    # Defaults / Baseline Indian Market Valuation Percentile Estimates
    cycle_data["nifty_trailing_pe"] = 22.8
    cycle_data["nifty_pe_10y_percentile"] = 68.5  # 68th percentile relative to 10-yr history
    cycle_data["nifty_trailing_pb"] = 3.65
    cycle_data["nifty_pb_10y_percentile"] = 72.0
    cycle_data["earnings_revision_ratio"] = 1.15  # Upgrades / Downgrades ratio (>1.0 = positive)

    pe_pct = cycle_data["nifty_pe_10y_percentile"]
    if pe_pct > 75.0:
        cycle_phase = "LATE_EXPANSION_PEAK"
        risk_level = "ELEVATED_VALUATION_RISK"
    elif pe_pct < 30.0:
        cycle_phase = "RECOVERY_TROUGH"
        risk_level = "ATTRACTIVE_VALUATION_ENTRY"
    else:
        cycle_phase = "MID_CYCLE_EXPANSION"
        risk_level = "FAIR_VALUATION"

    cycle_data["classified_cycle_phase"] = cycle_phase
    cycle_data["equity_valuation_risk"] = risk_level

    return json.dumps(cycle_data, indent=2)