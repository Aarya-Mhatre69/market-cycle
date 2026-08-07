"""
Pure ML Model MCP Server: Equity Models (Streamable HTTP)
Exposes LightGBM Quantile Regression Price Bands and KMeans/IsolationForest Clustering.

Port: 8003
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Optional, Dict

import joblib
import pandas as pd
import yfinance as yf
from fastmcp import FastMCP

from shankh.agents.equity.pb_config import CONFIG as PB_CONFIG
from shankh.agents.equity.pb_features import add_features
from shankh.agents.equity.pb_inference import load_models_and_features, predict_next_day_band

from shankh.agents.equity.cluster_config import CONFIG as CLUSTER_CONFIG
from shankh.agents.equity.cluster_features import build_cluster_features
from shankh.agents.equity.cluster_validation import validate_features as validate_cluster_features
from shankh.agents.equity.cluster_model import KMeansModel, IsolationForestModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_equity_ml_server")

_HISTORY_DAYS = 200
_PADDING_ANCHORS = [
    "HDFCBANK.NS", "INFY.NS", "AXISBANK.NS", "HINDUNILVR.NS", "DRREDDY.NS",
    "GAIL.NS", "ITC.NS", "CIPLA.NS", "BPCL.NS", "BRITANNIA.NS",
]


# ============================================================================
# 1. EAGER MODEL CONTAINER: PRICE BAND LIGHTGBM QUANTILE BOOSTERS
# ============================================================================
class PriceBandMLContainer:
    """Eagerly loads LightGBM boosters and Conformal Calibration scores into RAM."""

    def __init__(self):
        logger.info("Eagerly loading LightGBM Quantile Regression Boosters into RAM...")
        cfg_lgb = copy.deepcopy(PB_CONFIG)
        cfg_lgb["model"]["backend"] = "lightgbm"
        self.cfg_lgb = cfg_lgb

        # Unpack 4-tuple including Split Conformal Calibration score (q_conf)
        self.lgb_upper, self.lgb_lower, self.lgb_features, self.q_conf = (
            load_models_and_features(cfg_lgb)
        )
        logger.info(
            "Price Band boosters eagerly loaded. Features: %d | q_conf: %.5f",
            len(self.lgb_features),
            self.q_conf,
        )

    def run_inference(
        self, df_feat: pd.DataFrame, current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        row = df_feat.iloc[-1]
        latest_date = str(
            row["date"].strftime("%Y-%m-%d")
            if hasattr(row["date"], "strftime")
            else str(row["date"])
        )

        missing = set(self.lgb_features) - set(df_feat.columns)
        if missing:
            raise ValueError(f"Feature mismatch missing: {sorted(missing)}")

        feat_dict = {c: float(row[c]) for c in self.lgb_features}
        base_close = (
            float(current_price) if current_price and current_price > 0 else float(row["close"])
        )

        pred = predict_next_day_band(
            row=feat_dict,
            upper_model=self.lgb_upper,
            lower_model=self.lgb_lower,
            feature_cols=self.lgb_features,
            close=base_close,
        )

        pred_upper_ret = pred["pred_upper_ret"] + self.q_conf
        pred_lower_ret = pred["pred_lower_ret"] - self.q_conf

        pred_high = round(base_close * (1.0 + pred_upper_ret), 2)
        pred_low = round(base_close * (1.0 + pred_lower_ret), 2)
        band_width = round(((pred_high - pred_low) / base_close) * 100.0, 2)

        upper_q_label = int(self.cfg_lgb["model"].get("upper_quantile", 0.84) * 100)
        lower_q_label = int(self.cfg_lgb["model"].get("lower_quantile", 0.16) * 100)

        return {
            "latest_feature_date": latest_date,
            "model_backend": "lightgbm",
            "conformal_calibration_applied": True,
            "conformal_q_adjustment": round(self.q_conf, 5),
            "predicts_next_trading_day_band": True,
            "base_close_price_inr": round(base_close, 2),
            "lightgbm_predicted_price_band": {
                "predicted_high_price_inr": pred_high,
                "predicted_low_price_inr": pred_low,
                "expected_band_width_pct": f"{band_width}%",
                f"upper_return_quantile_{upper_q_label}th_pct": round(pred_upper_ret * 100, 4),
                f"lower_return_quantile_{lower_q_label}th_pct": round(pred_lower_ret * 100, 4),
            },
        }


# ============================================================================
# 2. EAGER MODEL CONTAINER: KMEANS & ISOLATION FOREST CLUSTERING
# ============================================================================
class EquityClusterMLContainer:
    """Eagerly loads KMeans Peer Clustering and IsolationForest models into RAM."""

    def __init__(self, models_dir: Path = None):
        models_dir = Path(models_dir or CLUSTER_CONFIG["artifacts"]["models_dir"])
        logger.info("Eagerly pre-loading Equity Clustering & Anomaly models into RAM...")

        scaler_path = models_dir / CLUSTER_CONFIG["artifacts"]["cluster_scaler"]
        kmeans_path = models_dir / CLUSTER_CONFIG["artifacts"]["kmeans_model"]
        iso_path = models_dir / CLUSTER_CONFIG["artifacts"]["isolation_forest"]

        if not scaler_path.exists() or not kmeans_path.exists() or not iso_path.exists():
            raise FileNotFoundError(f"Model artifacts missing in {models_dir}.")

        self.scaler = joblib.load(scaler_path)

        self.kmeans_model = KMeansModel(n_clusters=CLUSTER_CONFIG["model"]["n_clusters"])
        self.kmeans_model.load(kmeans_path)

        self.iso_forest = IsolationForestModel()
        self.iso_forest.load(iso_path)

        logger.info("Equity Clustering models eagerly loaded successfully.")

    def run_inference(self, df: pd.DataFrame) -> Dict[str, Any]:
        pivoted_prices, factor_df = build_cluster_features(df, CLUSTER_CONFIG["features"])
        if not validate_cluster_features(factor_df, pivoted_prices):
            raise ValueError("Feature validation failed during cluster inference.")

        if hasattr(self.scaler, "feature_names_in_"):
            factor_df = factor_df[list(self.scaler.feature_names_in_)]

        scaled_factors = self.scaler.transform(factor_df)

        kmeans_labels = self.kmeans_model.predict(scaled_factors)
        kmeans_assignments = {
            ticker: int(label) for ticker, label in zip(factor_df.index, kmeans_labels)
        }

        anomaly_preds = self.iso_forest.predict(scaled_factors)
        anomalies = [
            ticker for ticker, pred in zip(factor_df.index, anomaly_preds) if pred == -1
        ]

        return {
            "kmeans_factor_clusters": kmeans_assignments,
            "flagged_technical_outliers": anomalies,
        }


PRICE_BAND_CONTAINER = PriceBandMLContainer()
CLUSTER_CONTAINER = EquityClusterMLContainer()

# ============================================================================
# FAST MCP SERVER INITIALIZATION & PURE MODEL TOOL ENDPOINTS
# ============================================================================
mcp = FastMCP("Equity ML Model Server")


def _fetch_ohlcv_single(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=f"{_HISTORY_DAYS}d", interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No OHLCV data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = ticker
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].sort_values("date").reset_index(drop=True)


@mcp.tool()
def query_gbm_price_band(ticker: str, current_price: Optional[float] = None) -> str:
    """
    Pure ML Model Inference: Predict next-day high/low price band using LightGBM Quantile Boosters.

    Args:
        ticker: NSE ticker symbol (e.g., 'RELIANCE.NS', 'INFY.NS').
        current_price: Optional today close price override (INR).
    """
    ticker_clean = ticker.strip().upper()
    if not ticker_clean.endswith(".NS"):
        ticker_clean += ".NS"

    try:
        df_raw = _fetch_ohlcv_single(ticker_clean)
        cfg = PRICE_BAND_CONTAINER.cfg_lgb
        df_feat = add_features(df_raw, cfg=cfg.get("features"))
        if df_feat.empty:
            return json.dumps({"error": "Empty feature matrix generated."}, indent=2)

        res = PRICE_BAND_CONTAINER.run_inference(df_feat, current_price)
        res["ticker"] = ticker_clean
        return json.dumps(res, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


@mcp.tool()
def get_stock_clusters(tickers: str) -> str:
    """
    Pure ML Model Inference: Run KMeans Peer Clustering and IsolationForest Outlier Detection.

    Args:
        tickers: Comma-separated NSE tickers (e.g. 'TCS.NS,INFY.NS,RELIANCE.NS').
    """
    if not tickers or not tickers.strip():
        return json.dumps({"error": "tickers parameter is required."}, indent=2)

    requested = [(t.strip().upper() if t.strip().upper().endswith(".NS") else t.strip().upper() + ".NS") for t in tickers.split(",") if t.strip()]
    min_required = CLUSTER_CONFIG["data"]["min_tickers"]

    padding = [t for t in _PADDING_ANCHORS if t not in requested]
    shortfall = max(0, min_required - len(requested))
    fetch_list = requested + padding[:shortfall]

    frames = []
    for tkr in fetch_list:
        try:
            frames.append(_fetch_ohlcv_single(tkr))
        except Exception:
            pass

    if not frames:
        return json.dumps({"error": "Could not fetch OHLCV data for cluster tickers."}, indent=2)

    df_combined = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

    try:
        result = CLUSTER_CONTAINER.run_inference(df_combined)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    all_clusters = result.get("kmeans_factor_clusters", {})
    all_anomalies = result.get("flagged_technical_outliers", [])

    requested_clusters = {t: v for t, v in all_clusters.items() if t in requested}
    requested_anomalies = [t for t in all_anomalies if t in requested]

    cluster_to_members: Dict[int, list] = {}
    for t, cid in requested_clusters.items():
        cluster_to_members.setdefault(cid, []).append(t)

    ticker_analysis = {}
    for t, cid in requested_clusters.items():
        peers = [p for p in cluster_to_members[cid] if p != t]
        ticker_analysis[t] = {
            "cluster_id": cid,
            "peers_in_same_cluster": peers,
            "is_technical_outlier": t in requested_anomalies,
        }

    return json.dumps({
        "ticker_analysis": ticker_analysis,
        "flagged_technical_outliers": requested_anomalies,
    }, indent=2, default=str)


if __name__ == "__main__":
    logger.info("Starting Equity Pure ML Model MCP Server on Streamable HTTP (Port 8003)...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)