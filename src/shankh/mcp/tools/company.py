"""
Quantile Regression Price Band MCP Server (Streamable HTTP)
Port: 8003
"""
import copy
import json
import logging
from typing import Any, Optional

import pandas as pd
import yfinance as yf
from fastmcp import FastMCP

from shankh.agents.company.config import CONFIG
from shankh.agents.company.features import add_features
from shankh.agents.company.inference import load_models_and_features, predict_next_day_band

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_price_band_server")

_HISTORY_DAYS = 200

# ============================================================================
# EAGER MODEL LOADING CONTAINER
# ============================================================================
class EagerPriceBandContainer:
    def __init__(self):
        logger.info("Eagerly loading LightGBM Quantile Regression Boosters...")

        cfg_lgb = copy.deepcopy(CONFIG)
        cfg_lgb["model"]["backend"] = "lightgbm"
        self.cfg_lgb = cfg_lgb

        # Eager loading happens here once during server launch
        self.lgb_upper, self.lgb_lower, self.lgb_features = load_models_and_features(cfg_lgb)

        logger.info("Price Band boosters eagerly loaded into memory successfully.")

    def run_eager_inference(
        self,
        df_feat: pd.DataFrame,
        current_price: Optional[float] = None,
    ) -> dict[str, Any]:
        upper_m, lower_m, feats, cfg = self.lgb_upper, self.lgb_lower, self.lgb_features, self.cfg_lgb

        row = df_feat.iloc[-1]
        latest_feature_date = str(row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]))

        missing = set(feats) - set(df_feat.columns)
        if missing:
            raise ValueError(f"Feature mismatch missing: {sorted(missing)}")

        feat_dict = {c: float(row[c]) for c in feats}
        base_close = float(current_price) if current_price and current_price > 0 else float(row["close"])

        pred = predict_next_day_band(
            row=feat_dict,
            upper_model=upper_m,
            lower_model=lower_m,
            feature_cols=feats,
            close=base_close,
        )

        upper_q_label = int(cfg["model"].get("upper_quantile", 0.84) * 100)
        lower_q_label = int(cfg["model"].get("lower_quantile", 0.16) * 100)

        return {
            "latest_feature_date": latest_feature_date,
            "model_backend": "lightgbm",
            "predicts_next_trading_day_band": True,
            "base_close_price_inr": round(base_close, 2),
            "lightgbm_predicted_price_band": {
                "predicted_high_price_inr": round(pred["pred_high_price"], 2),
                "predicted_low_price_inr": round(pred["pred_low_price"], 2),
                "expected_band_width_pct": f"{round(pred['band_width_pct'], 2)}%",
                f"upper_return_quantile_{upper_q_label}th_pct": round(pred["pred_upper_ret"] * 100, 4),
                f"lower_return_quantile_{lower_q_label}th_pct": round(pred["pred_lower_ret"] * 100, 4),
            },
        }

CONTAINER = EagerPriceBandContainer()

# ============================================================================
# MCP SERVER INITIALIZATION & TOOLS
# ============================================================================
mcp = FastMCP("Price Band Server")

def _fetch_ohlcv(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=f"{_HISTORY_DAYS}d", interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = ticker
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].sort_values("date").reset_index(drop=True)

def _execute_query(ticker: str, current_price: Optional[float]) -> str:
    ticker_clean = ticker.strip().upper()
    if not ticker_clean.endswith(".NS"):
        ticker_clean += ".NS"

    try:
        df_raw = _fetch_ohlcv(ticker_clean)
        cfg = CONTAINER.cfg_lgb
        df_feat = add_features(df_raw, cfg=cfg.get("features"))
        if df_feat.empty:
            return json.dumps({"error": "Empty feature set generated."}, indent=2)

        res = CONTAINER.run_eager_inference(df_feat, current_price)
        res["ticker"] = ticker_clean
        return json.dumps(res, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

@mcp.tool()
def query_gbm_price_band(ticker: str, current_price: Optional[float] = None) -> str:
    """Predict next trading day price band for an NSE stock using LightGBM Quantile Regression."""
    return _execute_query(ticker, current_price)

if __name__ == "__main__":
    logger.info("Starting Price Band MCP Server on Streamable HTTP (Port 8003)...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)
