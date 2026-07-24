import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from shankh.agents.shared_tools import get_web_search_tool, filter_tools
from shankh.ml.market.tool import get_market_regime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MARKET_HISTORICAL_PATH = PROJECT_ROOT / "models" / "regime" / "historical_regimes.csv"
MARKET_METADATA_PATH = PROJECT_ROOT / "models" / "regime" / "regime_metadata.json"


def _load_market_artifacts() -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    if not MARKET_HISTORICAL_PATH.exists() or not MARKET_METADATA_PATH.exists():
        return None, None

    with open(MARKET_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df_hist = pd.read_csv(MARKET_HISTORICAL_PATH)
    df_hist["date"] = pd.to_datetime(df_hist["date"]).dt.strftime("%Y-%m-%d")
    return df_hist, metadata


@tool
def query_market_conditions(query_date: Optional[str] = None) -> str:
    """Query market state classification, volatility, breadth, and correlation density for a date.

    Args:
        query_date: Date in YYYY-MM-DD format. If None or latest, returns latest market state.
    """
    df_hist, metadata = _load_market_artifacts()
    if df_hist is None or metadata is None:
        return json.dumps({"error": "Market artifacts not found. Run 'train_regime' first."})

    if not query_date or query_date.lower() in ["latest", "today", "now"]:
        latest_row = df_hist.iloc[-1]
        return json.dumps({
            "date": latest_row["date"],
            "classified_market": str(latest_row["regime_label"]).upper(),
            "market_volatility_pct": round(float(latest_row["mkt_volatility"]), 2),
            "breadth_pct_above_20dma": round(float(latest_row["breadth_pct_above_20dma"]), 1),
            "correlation_density": round(float(latest_row["correlation_density"]), 3),
            "ad_ratio": round(float(latest_row.get("ad_ratio", 1.0)), 2),
            "historical_market_distribution": metadata.get("state_statistics"),
        }, indent=2)

    clean_date = query_date.strip()
    match = df_hist[df_hist["date"] == clean_date]

    if match.empty:
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
        "classified_market": str(row["regime_label"]).upper(),
        "market_volatility_pct": round(float(row["mkt_volatility"]), 2),
        "breadth_pct_above_20dma": round(float(row["breadth_pct_above_20dma"]), 1),
        "correlation_density": round(float(row["correlation_density"]), 3),
        "ad_ratio": round(float(row.get("ad_ratio", 1.0)), 2),
        "note": note,
    }, indent=2)


@tool
def query_market_regime(query_date: Optional[str] = None) -> str:
    """Backwards-compatible alias for querying market conditions by date."""
    return query_market_conditions.invoke({"query_date": query_date})


def get_market_analyst_tools() -> list:
    """Tools owned by the Market Analyst only."""
    return filter_tools([
        get_web_search_tool(),
        get_market_regime,
    ])
