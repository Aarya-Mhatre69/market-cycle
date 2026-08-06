"""
Equity Specialist Tools Module.

Exposes LangChain tools for live LightGBM Quantile Regression Price-Band
inference with Split Conformal Calibration adjustments.
"""

import json
import logging
from typing import Optional

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from shankh.agents.equity.pb_config import CONFIG
from shankh.agents.equity.pb_features import add_features
from shankh.agents.equity.pb_inference import (
    load_models_and_features,
    predict_next_day_band,
)

logger = logging.getLogger(__name__)

# 200 trading days warmup buffer for rolling 50-bar windows and regime features
_HISTORY_DAYS = 200


def _fetch_ohlcv(ticker: str, days: int = _HISTORY_DAYS) -> pd.DataFrame:
    """
    Download OHLCV from yfinance and return a clean DataFrame with columns:
    date, open, high, low, close, volume, ticker.
    """
    raw = yf.download(
        ticker,
        period=f"{days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for '{ticker}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = ticker
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df.sort_values("date").reset_index(drop=True)
    return df


@tool
def query_gbm_price_band(
    ticker: str,
    current_price: Optional[float] = None,
) -> str:
    """
    Predict the next trading day's high/low price band for an NSE stock using
    the live LightGBM Quantile Regression model with Conformal Prediction.

    The tool fetches recent OHLCV data from yfinance, engineers features on the
    latest trading day, loads the trained LightGBM quantile boosters and conformal
    calibration score, and projects the predicted price band for the NEXT trading day.

    Args:
        ticker: NSE ticker symbol (e.g. 'INFY.NS', 'TCS.NS', 'RELIANCE.NS').
        current_price: Optional override for today's close price (INR).
                       If supplied, price bands are projected relative to this
                       base instead of the latest historical close.

    Returns:
        JSON string with predicted next-day high/low prices, band width %,
        return quantiles, base close used, and latest feature date.
    """
    return _execute_price_band_query(
        ticker=ticker,
        current_price=current_price,
    )


def _execute_price_band_query(
    ticker: str,
    current_price: Optional[float],
) -> str:
    """Internal helper driving OHLCV retrieval, feature creation, and model inference."""
    cfg = CONFIG

    ticker_clean = ticker.strip().upper()
    if not ticker_clean.endswith(".NS"):
        ticker_clean += ".NS"

    try:
        df_raw = _fetch_ohlcv(ticker_clean, days=_HISTORY_DAYS)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    if len(df_raw) < 60:
        return json.dumps({
            "error": f"Insufficient history for '{ticker_clean}' "
                     f"({len(df_raw)} rows). Need at least 60 trading days."
        }, indent=2)

    try:
        df_feat = add_features(df_raw, cfg=cfg.get("features"))
    except Exception as exc:
        logger.exception("Feature engineering failed for %s", ticker_clean)
        return json.dumps({"error": f"Feature engineering failed: {exc}"}, indent=2)

    if df_feat.empty:
        return json.dumps({
            "error": "Feature engineering produced an empty frame. "
                     "The ticker may not have enough trading history."
        }, indent=2)

    row = df_feat.iloc[-1]
    latest_feature_date = (
        row["date"].strftime("%Y-%m-%d")
        if hasattr(row["date"], "strftime")
        else str(row["date"])
    )

    try:
        upper_model, lower_model, feature_cols, q_conf = load_models_and_features(cfg)
    except Exception as exc:
        logger.exception("Model loading failed")
        return json.dumps({
            "error": f"Failed to load trained LightGBM models: {exc}",
            "action_required": "Run training pipeline first: python -m shankh.agents.equity.train"
        }, indent=2)

    missing_feats = set(feature_cols) - set(df_feat.columns)
    if missing_feats:
        return json.dumps({
            "error": "Feature mismatch between live data and trained model.",
            "missing_features": sorted(missing_feats),
        }, indent=2)

    feat_dict = {c: float(row[c]) for c in feature_cols}
    base_close = float(current_price) if current_price and current_price > 0 else float(row["close"])

    try:
        pred = predict_next_day_band(
            row=feat_dict,
            upper_model=upper_model,
            lower_model=lower_model,
            feature_cols=feature_cols,
            close=base_close,
        )
    except Exception as exc:
        logger.exception("Inference failed for %s", ticker_clean)
        return json.dumps({"error": f"Model inference failed: {exc}"}, indent=2)

    # Apply Conformal Prediction adjustment (q_conf) to predicted return bounds
    pred_upper_ret = pred["pred_upper_ret"] + q_conf
    pred_lower_ret = pred["pred_lower_ret"] - q_conf

    pred_high_price = round(base_close * (1.0 + pred_upper_ret), 2)
    pred_low_price  = round(base_close * (1.0 + pred_lower_ret), 2)
    band_width_pct  = round(((pred_high_price - pred_low_price) / base_close) * 100.0, 2)

    upper_q_label = int(cfg["model"].get("upper_quantile", 0.84) * 100)
    lower_q_label = int(cfg["model"].get("lower_quantile", 0.16) * 100)

    return json.dumps({
        "ticker": ticker_clean,
        "latest_feature_date": latest_feature_date,
        "model_backend": "lightgbm",
        "conformal_calibration_applied": True,
        "conformal_q_adjustment": round(q_conf, 5),
        "predicts_next_trading_day_band": True,
        "base_close_price_inr": round(base_close, 2),
        "lightgbm_predicted_price_band": {
            "predicted_high_price_inr": pred_high_price,
            "predicted_low_price_inr":  pred_low_price,
            "expected_band_width_pct":  f"{band_width_pct}%",
            f"upper_return_quantile_{upper_q_label}th_pct": round(pred_upper_ret * 100, 4),
            f"lower_return_quantile_{lower_q_label}th_pct": round(pred_lower_ret * 100, 4),
        },
    }, indent=2)