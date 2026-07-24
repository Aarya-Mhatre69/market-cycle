"""
src/shankh/ml/price_band/tool.py

LangChain Tool for live XGBoost Quantile Regression Price-Band inference.

Flow
----
1.  Fetch OHLCV history from yfinance (enough bars to warm up all features).
2.  Run add_features() to build the full feature frame.
3.  Load trained upper / lower XGBoost models + feature_cols from disk.
4.  Call predict_next_day_band() on the target row.
5.  Return a JSON summary — no parquet lookup, no pre-computed predictions.
"""

import json
import logging
from typing import Optional

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from shankh.ml.price_band.config import CONFIG
from shankh.ml.price_band.features import add_features
from shankh.ml.price_band.inference import load_models_and_features, predict_next_day_band

logger = logging.getLogger(__name__)

# Feature engineering needs up to a 50-bar rolling window plus regime
# features that need another 50 bars on top; 200 trading days is a safe buffer.
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

    # yfinance ≥0.2 returns MultiIndex columns even for a single ticker
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


def _nearest_date_row(df: pd.DataFrame, target_date: str) -> pd.Series:
    """
    Return the row whose date is closest to *target_date*.
    Uses strict equality first; falls back to nearest past trading date.
    """
    df = df.copy()
    df["_date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    exact = df[df["_date_str"] == target_date]
    if not exact.empty:
        return exact.iloc[-1], target_date, "exact"

    target_dt = pd.to_datetime(target_date)
    past_rows = df[df["date"] <= target_dt]
    if past_rows.empty:
        # Fall back to nearest overall
        idx = (df["date"] - target_dt).abs().idxmin()
    else:
        idx = (past_rows["date"] - target_dt).abs().idxmin()

    row = df.loc[idx]
    actual_date = row["_date_str"]
    return row, actual_date, "nearest"


@tool
def query_xgboost_price_band(
    ticker: str,
    date: Optional[str] = None,
    current_price: Optional[float] = None,
) -> str:
    """
    Predict the next-day high/low price band for an NSE stock using the
    live XGBoost Quantile Regression model.

    The tool fetches recent OHLCV data from yfinance, engineers all features
    on the fly, loads the trained upper/lower XGBoost models, and runs live
    inference — no pre-computed lookup tables are used.

    Args:
        ticker: NSE ticker symbol (e.g. 'INFY.NS', 'TCS.NS', 'RELIANCE.NS').
        date: Optional date string 'YYYY-MM-DD'. Defaults to the latest
              available trading day. Non-trading dates fall back to the
              nearest prior trading day in the fetched history.
        current_price: Optional override for today's close price (INR).
                       If supplied, price bands are projected from this
                       base instead of the day's actual close.

    Returns:
        JSON string with predicted high/low prices, band width %, return
        quantiles, base close used, and model performance benchmarks.
    """
    # ------------------------------------------------------------------
    # 1. Normalize ticker
    # ------------------------------------------------------------------
    ticker_clean = ticker.strip().upper()
    if not ticker_clean.endswith(".NS"):
        ticker_clean += ".NS"

    # ------------------------------------------------------------------
    # 2. Fetch live OHLCV
    # ------------------------------------------------------------------
    try:
        df_raw = _fetch_ohlcv(ticker_clean, days=_HISTORY_DAYS)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    if len(df_raw) < 60:
        return json.dumps({
            "error": f"Insufficient history for '{ticker_clean}' "
                     f"({len(df_raw)} rows). Need at least 60 trading days."
        }, indent=2)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    try:
        df_feat = add_features(df_raw, cfg=CONFIG.get("features"))
    except Exception as exc:
        logger.exception("Feature engineering failed for %s", ticker_clean)
        return json.dumps({"error": f"Feature engineering failed: {exc}"}, indent=2)

    if df_feat.empty:
        return json.dumps({
            "error": "Feature engineering produced an empty frame. "
                     "The ticker may not have enough trading history."
        }, indent=2)

    # ------------------------------------------------------------------
    # 4. Select target row
    # ------------------------------------------------------------------
    if not date or date.lower() in ("latest", "today", "now", ""):
        row = df_feat.iloc[-1]
        prediction_date = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        date_note = f"Using latest available feature date: {prediction_date}."
    else:
        row, prediction_date, match_type = _nearest_date_row(df_feat, date.strip())
        if match_type == "exact":
            date_note = f"Exact trading date match: {prediction_date}."
        else:
            date_note = (
                f"Requested date '{date.strip()}' is a non-trading day or outside "
                f"the fetched window. Using nearest prior trading date: {prediction_date}."
            )

    # ------------------------------------------------------------------
    # 5. Load models + feature cols (cached by Python's module import cache)
    # ------------------------------------------------------------------
    try:
        upper_model, lower_model, feature_cols = load_models_and_features(CONFIG)
    except Exception as exc:
        logger.exception("Model loading failed")
        return json.dumps({
            "error": f"Failed to load trained models: {exc}",
            "action_required": "Run the training pipeline first: "
                               "python train_xgb.py"
        }, indent=2)

    # Verify all expected features are present
    missing_feats = set(feature_cols) - set(df_feat.columns)
    if missing_feats:
        return json.dumps({
            "error": "Feature mismatch between live data and trained model.",
            "missing_features": sorted(missing_feats),
        }, indent=2)

    # ------------------------------------------------------------------
    # 6. Build feature dict for the target row
    # ------------------------------------------------------------------
    feat_dict = {c: float(row[c]) for c in feature_cols}

    # Close used as the base for price-band projection
    base_close = float(current_price) if current_price and current_price > 0 else float(row["close"])

    # ------------------------------------------------------------------
    # 7. Inference
    # ------------------------------------------------------------------
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

    pred_high_price = round(pred["pred_high_price"], 2)
    pred_low_price  = round(pred["pred_low_price"],  2)
    pred_upper_ret  = pred["pred_upper_ret"]
    pred_lower_ret  = pred["pred_lower_ret"]
    band_width_pct  = round(pred["band_width_pct"], 2)

    # ------------------------------------------------------------------
    # 9. Build response
    # ------------------------------------------------------------------
    return json.dumps({
        "ticker": ticker_clean,
        "prediction_date": prediction_date,
        "predicts_next_trading_day_band": True,
        "base_close_price_inr": round(base_close, 2),
        "xgboost_predicted_price_band": {
            "predicted_high_price_inr": pred_high_price,
            "predicted_low_price_inr":  pred_low_price,
            "expected_band_width_pct":  f"{band_width_pct}%",
            "upper_return_quantile_84th_pct": round(pred_upper_ret * 100, 4),
            "lower_return_quantile_16th_pct": round(pred_lower_ret * 100, 4),
        },

        "note": date_note,
    }, indent=2)


# ---------------------------------------------------------------------------
# Quick smoke-test (run as script)
# ---------------------------------------------------------------------------
def main():
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"

    print(f"Running live price-band inference for {ticker} ...\n")
    result = query_xgboost_price_band.invoke({"ticker": ticker})
    print(json.dumps(json.loads(result), indent=2))


if __name__ == "__main__":
    main()
