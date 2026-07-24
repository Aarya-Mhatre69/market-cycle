"""
Direct tool invocation tests — no agent, no LLM.

Calls each ML model tool directly with real data:
  - tickers : INFY.NS, HDFCBANK.NS, AXISBANK.NS, DRREDDY.NS
  - date    : 2026-07-24

Run:
    uv run pytest tests/unit/test_tools.py -v -s
"""

import json
import logging
import pytest
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multiple tickers from the trained universe for richer cluster/peer context
TICKERS     = "INFY.NS,HDFCBANK.NS,AXISBANK.NS,DRREDDY.NS"
TICKER      = "INFY.NS"   # primary ticker for single-ticker tools
DATE        = "2026-07-24"


# ---------------------------------------------------------------------------
# price_band — query_xgboost_price_band (one ticker at a time by design)
# ---------------------------------------------------------------------------

def test_price_band_direct():
    from shankh.ml.price_band.tool import query_xgboost_price_band

    logger.info("Invoking query_xgboost_price_band(ticker=%s, date=%s)", TICKER, DATE)
    raw = query_xgboost_price_band.invoke({"ticker": TICKER, "date": DATE})
    result = json.loads(raw)
    logger.info("price_band result:\n%s", json.dumps(result, indent=2))

    if "error" in result and "action_required" in result:
        pytest.skip(f"Model artifacts not trained yet: {result['error']}")

    assert "xgboost_predicted_price_band" in result
    band = result["xgboost_predicted_price_band"]
    print(result)
    assert band["predicted_high_price_inr"] >= band["predicted_low_price_inr"]
    assert result["ticker"] == TICKER


# ---------------------------------------------------------------------------
# market — get_market_regime
# ---------------------------------------------------------------------------

def test_market_regime_direct():
    from shankh.ml.market.tool import get_market_regime

    logger.info("Invoking get_market_regime(query_date=%s)", DATE)
    raw = get_market_regime.invoke({"query_date": DATE})
    result = json.loads(raw)
    logger.info("market_regime result:\n%s", json.dumps(result, indent=2, default=str))
    print(result)

    assert "error" not in result, f"Tool returned error: {result.get('error')}"
    assert "regime_label" in result
    assert "volatility" in result
    assert "breadth_pct" in result


# ---------------------------------------------------------------------------
# macro — get_stock_clusters with multiple tickers
# ---------------------------------------------------------------------------

def test_stock_clusters_direct():
    from shankh.ml.macro.tool import get_stock_clusters

    logger.info("Invoking get_stock_clusters(tickers=%s)", TICKERS)
    raw = get_stock_clusters.invoke({"tickers": TICKERS})
    result = json.loads(raw)
    logger.info("stock_clusters result:\n%s", json.dumps(result, indent=2, default=str))
    print(result)

    assert "error" not in result, f"Tool returned error: {result.get('error')}"
    assert "ticker_analysis" in result
    assert "flagged_forensic_anomalies" in result

    for tkr in TICKERS.split(","):
        assert tkr.strip() in result["ticker_analysis"], \
            f"{tkr} not found in ticker_analysis"
        entry = result["ticker_analysis"][tkr.strip()]
        assert "cluster_id" in entry
        assert "peers_in_same_cluster" in entry
        assert "is_forensic_anomaly" in entry


# ---------------------------------------------------------------------------
# company — query_stock_peers (reads pre-trained artifact, single ticker)
# ---------------------------------------------------------------------------

def test_stock_peers_direct():
    from shankh.agents.company_tools import query_stock_peers

    logger.info("Invoking query_stock_peers(ticker=%s)", TICKER)
    raw = query_stock_peers.invoke({"ticker": TICKER})
    result = json.loads(raw)
    logger.info("stock_peers result:\n%s", json.dumps(result, indent=2))

    assert "error" not in result, f"Tool returned error: {result.get('error')}"
    assert result["ticker"] == TICKER
    assert "factor_cluster_id" in result
    assert "peer_group_sample" in result


# ---------------------------------------------------------------------------
# company — query_forensic_red_flags (reads pre-trained artifact, single ticker)
# ---------------------------------------------------------------------------

def test_forensic_red_flags_direct():
    from shankh.agents.company_tools import query_forensic_red_flags

    logger.info("Invoking query_forensic_red_flags(ticker=%s)", TICKER)
    raw = query_forensic_red_flags.invoke({"ticker": TICKER})
    result = json.loads(raw)
    logger.info("forensic_red_flags result:\n%s", json.dumps(result, indent=2))

    assert "error" not in result, f"Tool returned error: {result.get('error')}"
    assert result["ticker"] == TICKER
    assert "is_forensic_anomaly" in result
    assert "red_flag_summary" in result
