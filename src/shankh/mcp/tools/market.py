"""
Market Regime HMM MCP Server (Streamable HTTP)
Port: 8002
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import joblib
import yfinance as yf
from fastmcp import FastMCP

from shankh.agents.market.config import CONFIG
from shankh.agents.market.features import build_regime_features
from shankh.agents.market.validation import validate_features
from shankh.agents.market.model import HMMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_market_server")

_HISTORY_DAYS = 120
_NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
]

# ============================================================================
# EAGER MODEL LOADING CONTAINER
# ============================================================================
class MarketRegimeContainer:
    def __init__(self, models_dir: Path = None):
        models_dir = Path(models_dir or CONFIG["artifacts"]["models_dir"])
        logger.info("Eagerly loading Market Regime HMM model into RAM...")

        scaler_path = models_dir / CONFIG["artifacts"]["regime_scaler"]
        model_path = models_dir / CONFIG["artifacts"]["regime_model"]
        meta_path = models_dir / CONFIG["artifacts"]["regime_metadata"]

        if not scaler_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Missing regime model artifacts in {models_dir}")

        self.scaler = joblib.load(scaler_path)
        self.model = HMMModel(n_regimes=CONFIG["model"]["n_regimes"])
        self.model.load(model_path)

        self.state_mapping = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                raw_mapping = json.load(f).get("state_mapping", {})
                self.state_mapping = {int(k): str(v) for k, v in raw_mapping.items() if str(k).isdigit()}

        if not self.state_mapping:
            self.state_mapping = {i: f"State {i}" for i in range(CONFIG["model"]["n_regimes"])}

        logger.info("Market Regime HMM model eagerly loaded successfully.")

    def run_eager_inference(self, df: pd.DataFrame) -> Dict[str, Any]:
        regime_df = build_regime_features(df, CONFIG["features"])
        if not validate_features(regime_df):
            raise ValueError("Feature validation failed during inference.")

        feature_cols = getattr(
            self.scaler,
            "feature_names_in_",
            ["mkt_volatility", "breadth_pct_above_20dma", "correlation_density"]
        )

        X = self.scaler.transform(regime_df[list(feature_cols)])
        regime_states = self.model.predict(X)
        regime_df["regime_state"] = regime_states
        regime_df["regime_label"] = regime_df["regime_state"].map(self.state_mapping)

        latest_date = regime_df.index[-1]
        latest_date_str = str(latest_date.date()) if hasattr(latest_date, "date") else str(pd.to_datetime(latest_date).date())

        return {
            "date": latest_date_str,
            "regime_label": str(regime_df["regime_label"].iloc[-1]),
            "volatility": round(float(regime_df["mkt_volatility"].iloc[-1]), 2),
            "breadth_pct": round(float(regime_df["breadth_pct_above_20dma"].iloc[-1]), 2),
            "correlation_density": round(float(regime_df["correlation_density"].iloc[-1]), 2),
            "historical_regimes": {
                str(k.date()) if hasattr(k, "date") else str(k): v
                for k, v in regime_df[["regime_state", "regime_label"]].tail(10).to_dict(orient="index").items()
            }
        }

CONTAINER = MarketRegimeContainer()

# ============================================================================
# MCP SERVER INITIALIZATION & TOOL DEFINITION
# ============================================================================
mcp = FastMCP("Market Regime Server")

def _fetch_universe_ohlcv(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for tkr in tickers:
        try:
            raw = yf.download(tkr, period=f"{_HISTORY_DAYS}d", interval="1d", auto_adjust=False, progress=False)
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
            if len(df) >= 30:
                frames.append(df)
        except Exception:
            pass

    if not frames:
        raise RuntimeError("Could not fetch OHLCV data for universe.")
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

@mcp.tool()
def get_market_regime(query_date: Optional[str] = None) -> str:
    """
    Fetch the current Indian equity market regime using the live HMM model.

    Args:
        query_date: Optional date string 'YYYY-MM-DD'. Defaults to latest trading day.
    """
    try:
        df_raw = _fetch_universe_ohlcv(_NIFTY50_TICKERS)
        result = CONTAINER.run_eager_inference(df_raw)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    if query_date and query_date.lower() not in ("latest", "today", "now", ""):
        hist = result.get("historical_regimes", {})
        matching = {k: v for k, v in hist.items() if str(k).startswith(query_date)}
        if matching:
            result["historical_regimes"] = matching
            result["note"] = f"Showing regime data for requested date: {query_date}"

    return json.dumps(result, indent=2, default=str)

if __name__ == "__main__":
    logger.info("Starting Market Regime MCP Server on Streamable HTTP (Port 8002)...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8002)