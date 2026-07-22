"""
src/shankh/tools/analytics_tools.py

LangChain Tools for querying trained Stock Clustering, Forensic Anomalies,
and Market Regime time-series models.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Paths to trained model artifacts
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGIME_HISTORICAL_PATH = PROJECT_ROOT / "models" / "regime" / "historical_regimes.csv"
REGIME_METADATA_PATH = PROJECT_ROOT / "models" / "regime" / "regime_metadata.json"


# ---------------------------------------------------------------------------
# Tool 1: Market Regime Inference Tool (With Date Query Support)
# ---------------------------------------------------------------------------
@tool
def query_market_regime(query_date: Optional[str] = None) -> str:
    """
    Query market regime classification, volatility, breadth, and correlation density for a date.

    Args:
        query_date: Target date string in 'YYYY-MM-DD' format (e.g. '2024-02-20'). 
                    If None or 'latest', returns the active regime for the latest date.

    Returns:
        JSON formatted string containing market regime metrics and classification.
    """
    if not REGIME_HISTORICAL_PATH.exists() or not REGIME_METADATA_PATH.exists():
        return json.dumps({
            "error": "Regime artifacts not found. Run 'train_regime' first."
        })

    with open(REGIME_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_hist = pd.read_csv(REGIME_HISTORICAL_PATH)
    df_hist["date"] = pd.to_datetime(df_hist["date"]).dt.strftime("%Y-%m-%d")

    # Scenario A: Latest date lookup
    if not query_date or query_date.lower() in ["latest", "today", "now"]:
        latest_row = df_hist.iloc[-1]
        return json.dumps({
            "date": latest_row["date"],
            "classified_regime": str(latest_row["regime_label"]).upper(),
            "market_volatility_pct": round(float(latest_row["mkt_volatility"]), 2),
            "breadth_pct_above_20dma": round(float(latest_row["breadth_pct_above_20dma"]), 1),
            "correlation_density": round(float(latest_row["correlation_density"]), 3),
            "ad_ratio": round(float(latest_row.get("ad_ratio", 1.0)), 2),
            "historical_regime_distribution": metadata.get("state_statistics")
        }, indent=2)

    # Scenario B: Specific date lookup (with fallback to nearest trading date)
    clean_date = query_date.strip()
    match = df_hist[df_hist["date"] == clean_date]

    if match.empty:
        # Nearest trading date lookup
        try:
            target_dt = pd.to_datetime(clean_date)
            df_hist["dt_obj"] = pd.to_datetime(df_hist["date"])
            nearest_idx = (df_hist["dt_obj"] - target_dt).abs().idxmin()
            row = df_hist.loc[nearest_idx]
            note = f"Exact date '{clean_date}' was a non-trading day. Showing nearest date '{row['date']}'."
        except Exception:
            return json.dumps({"error": f"Invalid date format '{clean_date}'. Use 'YYYY-MM-DD'."})
    else:
        row = match.iloc[0]
        note = "Exact trading date match found."

    return json.dumps({
        "queried_date": clean_date,
        "actual_date": row["date"],
        "classified_regime": str(row["regime_label"]).upper(),
        "market_volatility_pct": round(float(row["mkt_volatility"]), 2),
        "breadth_pct_above_20dma": round(float(row["breadth_pct_above_20dma"]), 1),
        "correlation_density": round(float(row["correlation_density"]), 3),
        "ad_ratio": round(float(row.get("ad_ratio", 1.0)), 2),
        "note": note,
    }, indent=2)