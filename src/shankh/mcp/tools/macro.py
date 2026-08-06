"""
Macro Stock Clustering & Anomaly MCP Server (Streamable HTTP)
Port: 8001
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import joblib
import yfinance as yf
from fastmcp import FastMCP

from shankh.agents.equity.cluster_config import CONFIG
from shankh.agents.equity.cluster_features import build_cluster_features
from shankh.agents.equity.cluster_validation import validate_features
from shankh.agents.equity.cluster_model import KMeansModel, IsolationForestModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp_macro_server")

_HISTORY_DAYS = 120
_PADDING_ANCHORS = [
    "HDFCBANK.NS", "INFY.NS", "AXISBANK.NS", "HINDUNILVR.NS", "DRREDDY.NS",
    "GAIL.NS", "ITC.NS", "CIPLA.NS", "BPCL.NS", "BRITANNIA.NS",
    "AUROPHARMA.NS", "HINDALCO.NS", "EICHERMOT.NS", "FEDERALBNK.NS", "IOC.NS",
]

# ============================================================================
# EAGER MODEL LOADING CONTAINER (Zero-Latency Inference)
# ============================================================================
class MacroModelContainer:
    def __init__(self, models_dir: Path = None):
        models_dir = Path(models_dir or CONFIG["artifacts"]["models_dir"])
        logger.info("Eagerly pre-loading Macro Clustering & Anomaly models into RAM...")

        scaler_path = models_dir / CONFIG["artifacts"]["cluster_scaler"]
        kmeans_path = models_dir / CONFIG["artifacts"]["kmeans_model"]
        iso_path = models_dir / CONFIG["artifacts"]["isolation_forest"]

        if not scaler_path.exists() or not kmeans_path.exists() or not iso_path.exists():
            raise FileNotFoundError(
                f"Model artifacts missing in {models_dir}. Train models before running server."
            )

        self.scaler = joblib.load(scaler_path)

        self.kmeans_model = KMeansModel(n_clusters=CONFIG["model"]["n_clusters"])
        self.kmeans_model.load(kmeans_path)

        self.iso_forest = IsolationForestModel()
        self.iso_forest.load(iso_path)

        logger.info("Macro models eagerly loaded into memory successfully.")

    def run_inference(self, df: pd.DataFrame) -> Dict[str, Any]:
        pivoted_prices, factor_df = build_cluster_features(df, CONFIG["features"])
        if not validate_features(factor_df, pivoted_prices):
            raise ValueError("Feature validation failed during inference.")

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
            "flagged_forensic_anomalies": anomalies,
        }

# Instantiate eagerly at server startup
CONTAINER = MacroModelContainer()

# ============================================================================
# MCP SERVER INITIALIZATION & TOOL DEFINITION
# ============================================================================
mcp = FastMCP("Macro Stock Clustering Server")

def _normalize(ticker: str) -> str:
    t = ticker.strip().upper()
    return t if t.endswith(".NS") else t + ".NS"

def _fetch_ohlcv(tickers: list[str], days: int = _HISTORY_DAYS) -> pd.DataFrame:
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
            if len(df) >= 60:
                frames.append(df)
        except Exception as exc:
            logger.warning("Fetch error for %s: %s", tkr, exc)

    if not frames:
        raise RuntimeError("Could not fetch OHLCV data.")
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

@mcp.tool()
def get_stock_clusters(tickers: str) -> str:
    """
    Run live stock clustering and forensic anomaly detection for any NSE stock(s).

    Args:
        tickers: Comma-separated NSE tickers, e.g. 'TCS.NS' or 'INFY,TCS,RELIANCE'.
    """
    if not tickers or not tickers.strip():
        return json.dumps({"error": "tickers parameter is required."}, indent=2)

    requested = [_normalize(t) for t in tickers.split(",") if t.strip()]
    min_required = CONFIG["data"]["min_tickers"]

    padding = [t for t in _PADDING_ANCHORS if t not in requested]
    shortfall = max(0, min_required - len(requested))
    fetch_list = requested + padding[:shortfall]

    try:
        df_raw = _fetch_ohlcv(fetch_list)
        result = CONTAINER.run_inference(df_raw)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    all_clusters = result.get("kmeans_factor_clusters", {})
    all_anomalies = result.get("flagged_forensic_anomalies", [])

    requested_clusters = {t: v for t, v in all_clusters.items() if t in requested}
    requested_anomalies = [t for t in all_anomalies if t in requested]

    cluster_to_members: dict[int, list[str]] = {}
    for t, cid in requested_clusters.items():
        cluster_to_members.setdefault(cid, []).append(t)

    ticker_analysis = {}
    for t, cid in requested_clusters.items():
        peers = [p for p in cluster_to_members[cid] if p != t]
        ticker_analysis[t] = {
            "cluster_id": cid,
            "peers_in_same_cluster": peers,
            "is_forensic_anomaly": t in requested_anomalies,
        }

    return json.dumps({
        "ticker_analysis": ticker_analysis,
        "flagged_forensic_anomalies": requested_anomalies,
    }, indent=2, default=str)

if __name__ == "__main__":
    logger.info("Starting Macro Clustering MCP Server on Streamable HTTP (Port 8001)...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)