"""
src/shankh/ml/macro/tool.py

LangChain Tool for live Stock Clustering + Forensic Anomaly inference.

Flow
----
1.  Accept tickers from the agent (required).
2.  Pad with a minimal set of liquid NSE anchors only if needed to satisfy
    the model's minimum sample count — anchors are never shown in output.
3.  Fetch OHLCV from yfinance for all tickers.
4.  Build per-ticker factor features (return, volatility, momentum, trend, beta).
5.  Load trained KMeans scaler + model and IsolationForest model from disk.
6.  Run inference on the padded set.
7.  Return results filtered to only the requested tickers.
"""

import json
import logging

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from shankh.ml.macro.config import CONFIG
from shankh.ml.macro.inference import run_inference

logger = logging.getLogger(__name__)

_HISTORY_DAYS = 120

# Minimal liquid anchors used ONLY as silent padding when the agent provides
# fewer tickers than the model minimum. Sourced from the trained universe.
_PADDING_ANCHORS = [
    "HDFCBANK.NS", "INFY.NS", "AXISBANK.NS", "HINDUNILVR.NS", "DRREDDY.NS",
    "GAIL.NS", "ITC.NS", "CIPLA.NS", "BPCL.NS", "BRITANNIA.NS",
    "AUROPHARMA.NS", "HINDALCO.NS", "EICHERMOT.NS", "FEDERALBNK.NS", "IOC.NS",
]


def _normalize(ticker: str) -> str:
    t = ticker.strip().upper()
    return t if t.endswith(".NS") else t + ".NS"


def _fetch_ohlcv(tickers: list[str], days: int = _HISTORY_DAYS) -> pd.DataFrame:
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
            if len(df) >= 60:
                frames.append(df)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", tkr, exc)

    if not frames:
        raise RuntimeError("Could not fetch OHLCV data for any of the requested tickers.")

    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


@tool
def get_stock_clusters(tickers: str) -> str:
    """
    Run live stock clustering and forensic anomaly detection for any NSE stock(s).

    Fetches recent OHLCV data from yfinance for the specified tickers, computes
    per-ticker factor features (annualised return, volatility, Sharpe, beta,
    RSI, ATR, max drawdown, trend distance), loads the trained KMeans and
    IsolationForest models, and returns cluster assignments and forensic flags.

    Args:
        tickers: Comma-separated NSE tickers, e.g. 'TCS.NS' or
                 'INFY,TCS,RELIANCE,HDFCBANK'. The .NS suffix is optional.

    Returns:
        JSON with:
          - kmeans_factor_clusters: {ticker: cluster_id}
          - flagged_forensic_anomalies: [tickers with unusual factor profiles]
          - note: padding info if anchors were added to satisfy model minimum
    """
    if not tickers or not tickers.strip():
        return json.dumps({"error": "tickers is required. Provide at least one NSE ticker, e.g. 'TCS.NS'."}, indent=2)

    requested = [_normalize(t) for t in tickers.split(",") if t.strip()]
    if not requested:
        return json.dumps({"error": "No valid tickers parsed. Use comma-separated NSE symbols."}, indent=2)

    min_required = CONFIG["data"]["min_tickers"]

    # Pad silently with anchors if below model minimum — exclude already requested
    padding = [t for t in _PADDING_ANCHORS if t not in requested]
    shortfall = max(0, min_required - len(requested))
    fetch_list = requested + padding[:shortfall]
    padded = shortfall > 0

    try:
        df_raw = _fetch_ohlcv(fetch_list, days=_HISTORY_DAYS)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    n_fetched = df_raw["ticker"].nunique()
    if n_fetched < min_required:
        return json.dumps({
            "error": (
                f"Only {n_fetched} tickers returned sufficient data after padding "
                f"(need {min_required}). Try providing more tickers."
            )
        }, indent=2)

    try:
        result = run_inference(df_raw)
    except FileNotFoundError as exc:
        return json.dumps({
            "error": str(exc),
            "action_required": "Train the clustering model first: uv run src/shankh/ml/macro/train_cluster.py",
        }, indent=2)
    except Exception as exc:
        logger.exception("Stock clustering inference failed")
        return json.dumps({"error": f"Inference failed: {exc}"}, indent=2)

    # Filter to requested tickers and enrich with peer context.
    # Peers are derived entirely from the live inference — no file lookup,
    # works identically in dev and prod.
    all_clusters: dict = result.get("kmeans_factor_clusters", {})
    all_anomalies: list = result.get("flagged_forensic_anomalies", [])

    requested_clusters = {t: v for t, v in all_clusters.items() if t in requested}
    requested_anomalies = [t for t in all_anomalies if t in requested]

    # Group requested tickers by cluster so each ticker can see its peers
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

    if padded:
        logger.info(
            "Clustering padded: %d anchor(s) added for model minimum of %d. "
            "Results filtered to: %s", shortfall, min_required, requested
        )

    return json.dumps({
        "ticker_analysis": ticker_analysis,
        "flagged_forensic_anomalies": requested_anomalies,
    }, indent=2, default=str)
