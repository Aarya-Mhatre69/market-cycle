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
CLUSTER_ARTIFACT_PATH = PROJECT_ROOT / "models" / "clustering" / "cluster_results.json"


# ---------------------------------------------------------------------------
# Tool 1: Stock Clustering & Forensic Anomaly Inference Tool
# ---------------------------------------------------------------------------
@tool
def query_stock_clustering_and_forensics(
    ticker: Optional[str] = None,
    cluster_id: Optional[int] = None,
    check_forensic_anomalies_only: bool = False,
) -> str:
    """
    Query stock clustering assignments, peer groupings, or forensic red flags.

    Args:
        ticker: Optional NSE ticker symbol (e.g. 'INFY.NS', 'ALOKINDS.NS').
        cluster_id: Optional cluster integer ID (e.g. 0, 1, 2) to list all members.
        check_forensic_anomalies_only: If True, returns only flagged forensic red-flag stocks.

    Returns:
        JSON formatted string containing clustering and forensic analysis.
    """
    if not CLUSTER_ARTIFACT_PATH.exists():
        return json.dumps({
            "error": "Clustering artifacts not found. Run 'train_cluster' first."
        })

    with open(CLUSTER_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    kmeans_clusters = artifacts.get("kmeans_factor_clusters", {})
    corr_clusters = artifacts.get("correlation_clusters", {})
    forensic_anomalies = artifacts.get("flagged_forensic_anomalies", [])

    # Scenario A: Check forensic anomalies only
    if check_forensic_anomalies_only:
        return json.dumps({
            "total_screened": len(kmeans_clusters),
            "flagged_forensic_anomalies_count": len(forensic_anomalies),
            "flagged_tickers": forensic_anomalies,
            "note": "These stocks exhibit extreme volatility, drawdown, or factor footprints requiring review."
        }, indent=2)

    # Scenario B: Query specific Ticker
    if ticker:
        ticker_clean = ticker.strip().upper()
        if not ticker_clean.endswith(".NS"):
            ticker_clean += ".NS"

        if ticker_clean not in kmeans_clusters:
            return json.dumps({
                "error": f"Ticker '{ticker_clean}' not found in trained universe.",
                "available_tickers_sample": list(kmeans_clusters.keys())[:5]
            })

        c_id = kmeans_clusters[ticker_clean]
        peers = [t for t, cid in kmeans_clusters.items() if cid == c_id and t != ticker_clean]
        is_anomaly = ticker_clean in forensic_anomalies

        return json.dumps({
            "ticker": ticker_clean,
            "factor_cluster_id": c_id,
            "correlation_cluster_id": corr_clusters.get(ticker_clean),
            "is_forensic_anomaly": is_anomaly,
            "peer_group_sample": peers[:8],
            "total_peers_count": len(peers),
        }, indent=2)

    # Scenario C: Query specific Cluster ID
    if cluster_id is not None:
        members = [t for t, cid in kmeans_clusters.items() if cid == cluster_id]
        if not members:
            return json.dumps({"error": f"No stocks found for cluster_id {cluster_id}."})

        return json.dumps({
            "cluster_id": cluster_id,
            "member_count": len(members),
            "members": members,
            "forensic_anomalies_in_cluster": [t for t in members if t in forensic_anomalies]
        }, indent=2)

    # Default: Return overall summary
    return json.dumps({
        "total_tickers": len(kmeans_clusters),
        "cluster_counts": pd.Series(list(kmeans_clusters.values())).value_counts().to_dict(),
        "forensic_anomalies": forensic_anomalies,
    }, indent=2)

