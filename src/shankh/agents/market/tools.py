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

from shankh.agents.market.inference import run_inference

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


def _fetch_universe_ohlcv(tickers: list[str], days: int = _HISTORY_DAYS) -> pd.DataFrame:
    """Download OHLCV for all tickers, return stacked DataFrame (date, open, high, low, close, volume, ticker)."""
    frames = []
    for tkr in tickers:
        try:
            raw = yf.download(tkr, period=f"{days}d", interval="1d", auto_adjust=False, progress=False)
            if raw.empty:
                logger.warning("No data for %s — skipping.", tkr)
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
        raise RuntimeError("Could not fetch OHLCV data for any ticker in the universe.")

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


@tool
def get_market_regime(query_date: Optional[str] = None) -> str:
    """
    Fetch the current Indian equity market regime using the live HMM model.

    Downloads recent OHLCV data for Nifty 50 stocks from yfinance, computes
    market-wide regime features (volatility, breadth, correlation density),
    loads the trained HMM scaler and model, and returns the current regime label.

    Args:
        query_date: Optional date string 'YYYY-MM-DD'. Defaults to the latest
                    available trading day.

    Returns:
        JSON with regime_label, volatility, breadth_pct, correlation_density,
        and a short history of recent regime states.
    """
    # 1. Fetch live OHLCV universe
    try:
        df_raw = _fetch_universe_ohlcv(_NIFTY50_TICKERS, days=_HISTORY_DAYS)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    if df_raw["ticker"].nunique() < 5:
        return json.dumps({"error": "Fewer than 5 tickers fetched — insufficient for regime signal."}, indent=2)

    # 2. Run inference (builds features + loads artifacts internally)
    try:
        result = run_inference(df_raw)
    except FileNotFoundError as exc:
        return json.dumps({
            "error": str(exc),
            "action_required": "Train the regime model first: uv run src/shankh/ml/market/train_regime.py",
        }, indent=2)
    except Exception as exc:
        logger.exception("Market regime inference failed")
        return json.dumps({"error": f"Inference failed: {exc}"}, indent=2)

    # 3. Optionally annotate for a specific date
    if query_date and query_date.lower() not in ("latest", "today", "now", ""):
        hist = result.get("historical_regimes", {})
        matching = {k: v for k, v in hist.items() if str(k).startswith(query_date)}
        if matching:
            result["historical_regimes"] = matching
            result["note"] = f"Showing regime data for requested date: {query_date}"
        else:
            result["note"] = (
                f"No exact match for '{query_date}' in fetched window. "
                f"Showing latest regime as of {result.get('date')}."
            )

    return json.dumps(result, indent=2, default=str)
