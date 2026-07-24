import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from shankh.ml.price_band.tool import query_xgboost_price_band
from shankh.agents.shared_tools import get_web_search_tool, filter_tools

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLUSTER_ARTIFACT_PATH = PROJECT_ROOT / "models" / "clustering" / "cluster_results.json"


def _load_cluster_artifacts() -> dict | None:
    if not CLUSTER_ARTIFACT_PATH.exists():
        return None

    with open(CLUSTER_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_ticker(ticker: str) -> str:
    ticker_clean = ticker.strip().upper()
    if not ticker_clean.endswith(".NS"):
        ticker_clean += ".NS"
    return ticker_clean


@tool
def query_stock_peers(ticker: Optional[str] = None, cluster_id: Optional[int] = None) -> str:
    """Query peer group, factor cluster, and correlation cluster for stocks.

    Args:
        ticker: Optional NSE ticker, such as INFY or INFY.NS.
        cluster_id: Optional factor cluster id to list all members.
    """
    artifacts = _load_cluster_artifacts()
    if artifacts is None:
        return json.dumps({"error": "Clustering artifacts not found. Run 'train_cluster' first."})

    kmeans_clusters = artifacts.get("kmeans_factor_clusters", {})
    corr_clusters = artifacts.get("correlation_clusters", {})
    forensic_anomalies = artifacts.get("flagged_forensic_anomalies", [])

    if ticker:
        ticker_clean = _normalize_ticker(ticker)
        if ticker_clean not in kmeans_clusters:
            return json.dumps({
                "error": f"Ticker '{ticker_clean}' not found in trained universe.",
                "available_tickers_sample": list(kmeans_clusters.keys())[:5],
            })

        factor_cluster_id = kmeans_clusters[ticker_clean]
        peers = [
            peer for peer, peer_cluster_id in kmeans_clusters.items()
            if peer_cluster_id == factor_cluster_id and peer != ticker_clean
        ]
        return json.dumps({
            "ticker": ticker_clean,
            "factor_cluster_id": factor_cluster_id,
            "correlation_cluster_id": corr_clusters.get(ticker_clean),
            "is_forensic_anomaly": ticker_clean in forensic_anomalies,
            "peer_group_sample": peers[:8],
            "total_peers_count": len(peers),
        }, indent=2)

    if cluster_id is not None:
        members = [ticker_symbol for ticker_symbol, cid in kmeans_clusters.items() if cid == cluster_id]
        if not members:
            return json.dumps({"error": f"No stocks found for cluster_id {cluster_id}."})

        return json.dumps({
            "cluster_id": cluster_id,
            "member_count": len(members),
            "members": members,
            "forensic_anomalies_in_cluster": [
                ticker_symbol for ticker_symbol in members if ticker_symbol in forensic_anomalies
            ],
        }, indent=2)

    return json.dumps({
        "total_tickers": len(kmeans_clusters),
        "cluster_counts": pd.Series(list(kmeans_clusters.values())).value_counts().to_dict(),
    }, indent=2)


@tool
def query_forensic_red_flags(ticker: Optional[str] = None) -> str:
    """Query forensic anomaly flags and red-flag summary for a stock or the full universe.

    Args:
        ticker: Optional NSE ticker, such as ALOKINDS or ALOKINDS.NS.
    """
    artifacts = _load_cluster_artifacts()
    if artifacts is None:
        return json.dumps({"error": "Clustering artifacts not found. Run 'train_cluster' first."})

    kmeans_clusters = artifacts.get("kmeans_factor_clusters", {})
    forensic_anomalies = artifacts.get("flagged_forensic_anomalies", [])

    if ticker:
        ticker_clean = _normalize_ticker(ticker)
        if ticker_clean not in kmeans_clusters:
            return json.dumps({
                "error": f"Ticker '{ticker_clean}' not found in trained universe.",
                "available_tickers_sample": list(kmeans_clusters.keys())[:5],
            })

        return json.dumps({
            "ticker": ticker_clean,
            "is_forensic_anomaly": ticker_clean in forensic_anomalies,
            "red_flag_summary": (
                "Flagged for extreme volatility, drawdown, or factor footprints requiring review."
                if ticker_clean in forensic_anomalies
                else "No forensic anomaly flag found in the trained artifact."
            ),
        }, indent=2)

    return json.dumps({
        "total_screened": len(kmeans_clusters),
        "flagged_forensic_anomalies_count": len(forensic_anomalies),
        "flagged_tickers": forensic_anomalies,
        "note": "These stocks exhibit extreme volatility, drawdown, or factor footprints requiring review.",
    }, indent=2)


@tool
def query_stock_clustering_and_forensics(
    ticker: Optional[str] = None,
    cluster_id: Optional[int] = None,
    check_forensic_anomalies_only: bool = False,
) -> str:
    """Backwards-compatible combined stock peer and forensic lookup."""
    if check_forensic_anomalies_only:
        return query_forensic_red_flags.invoke({"ticker": ticker})

    peer_result = json.loads(query_stock_peers.invoke({"ticker": ticker, "cluster_id": cluster_id}))
    if ticker and "error" not in peer_result:
        flags = json.loads(query_forensic_red_flags.invoke({"ticker": ticker}))
        peer_result["red_flag_summary"] = flags.get("red_flag_summary")
    elif not ticker and cluster_id is None:
        artifacts = _load_cluster_artifacts()
        forensic_anomalies = artifacts.get("flagged_forensic_anomalies", []) if artifacts else []
        peer_result["forensic_anomalies"] = forensic_anomalies
    return json.dumps(peer_result, indent=2)


def get_company_analyst_tools() -> list:
    """Tools owned by the Company Analyst only."""
    return filter_tools([
        get_web_search_tool(),
        query_xgboost_price_band,
        query_stock_peers,
        query_forensic_red_flags,
    ])
