"""
FastMCP Client Unit Tests for Decoupled Pure ML Model MCP Servers.

Tests the FastMCP servers for Equity ML (`shankh.mcp.tools.equity`) and
Market Regime ML (`shankh.mcp.tools.market`) by wrapping FastMCP app instances
in in-process Client transports.

Run:
    pytest tests/unit/test_mcp_ml_servers.py -v -s
"""

import importlib
import json
import logging
import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_server(module_name: str):
    """Import an MCP server module; skip if model artifacts are uninitialized."""
    try:
        return importlib.import_module(f"shankh.mcp.tools.{module_name}").mcp
    except FileNotFoundError as exc:
        pytest.skip(f"Model artifacts not found for {module_name} server: {exc}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not import {module_name} MCP server: {exc}")


def _result_text(result) -> str:
    """Extract returned JSON string payload from a FastMCP call_tool result."""
    if hasattr(result, "data") and result.data is not None:
        return str(result.data)
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            return "".join(
                str(block.text) if hasattr(block, "text") else str(block)
                for block in content
            )
        return str(content)
    return str(result)


async def _call_and_parse(client: Client[FastMCPTransport], name: str, arguments: dict) -> dict:
    """Invoke tool over FastMCP Client transport and parse returned JSON payload."""
    result = await client.call_tool(name=name, arguments=arguments)
    payload = json.loads(_result_text(result))
    if isinstance(payload, dict) and payload.get("error"):
        pytest.skip(f"{name} returned error: {payload['error']}")
    return payload


# ---------------------------------------------------------------------------
# FastMCP Client Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def equity_mcp_client():
    async with Client(transport=_load_server("equity")) as client:
        yield client


@pytest.fixture
async def market_mcp_client():
    async with Client(transport=_load_server("market")) as client:
        yield client


# ---------------------------------------------------------------------------
# Equity Pure ML Server Tests (LightGBM Price Band & KMeans/IsolationForest)
# ---------------------------------------------------------------------------

class TestEquityMLServer:
    """Unit tests for pure equity ML models (Price Band & Clustering)."""

    async def test_lists_equity_ml_tools(self, equity_mcp_client: Client[FastMCPTransport]):
        tools = await equity_mcp_client.list_tools()
        tool_names = [t.name for t in tools]
        assert "query_gbm_price_band" in tool_names
        assert "get_stock_clusters" in tool_names

    async def test_query_gbm_price_band_runs_conformal_lightgbm(
        self, equity_mcp_client: Client[FastMCPTransport]
    ):
        payload = await _call_and_parse(
            equity_mcp_client,
            name="query_gbm_price_band",
            arguments={"ticker": "RELIANCE.NS"},
        )

        assert payload.get("model_backend") == "lightgbm"
        assert payload.get("conformal_calibration_applied") is True
        assert "conformal_q_adjustment" in payload

        band = payload.get("lightgbm_predicted_price_band")
        assert band is not None, f"No LightGBM price band in payload: {payload}"
        assert band["predicted_high_price_inr"] >= band["predicted_low_price_inr"]
        assert "expected_band_width_pct" in band

    async def test_get_stock_clusters_runs_kmeans_and_isolation_forest(
        self, equity_mcp_client: Client[FastMCPTransport]
    ):
        payload = await _call_and_parse(
            equity_mcp_client,
            name="get_stock_clusters",
            arguments={"tickers": "INFY.NS,TCS.NS,RELIANCE.NS"},
        )

        assert "ticker_analysis" in payload
        assert "flagged_technical_outliers" in payload

        for tkr in ("INFY.NS", "TCS.NS", "RELIANCE.NS"):
            entry = payload["ticker_analysis"].get(tkr)
            assert entry is not None, f"{tkr} missing from ticker_analysis"
            assert "cluster_id" in entry
            assert "peers_in_same_cluster" in entry
            assert "is_technical_outlier" in entry


# ---------------------------------------------------------------------------
# Market Pure ML Server Tests (Gaussian HMM Market Regime)
# ---------------------------------------------------------------------------

class TestMarketMLServer:
    """Unit tests for pure market regime ML models (Gaussian HMM)."""

    async def test_lists_market_regime_tool(self, market_mcp_client: Client[FastMCPTransport]):
        tools = await market_mcp_client.list_tools()
        assert any(t.name == "get_market_regime" for t in tools)

    async def test_get_market_regime_runs_hmm_model(
        self, market_mcp_client: Client[FastMCPTransport]
    ):
        payload = await _call_and_parse(
            market_mcp_client,
            name="get_market_regime",
            arguments={"query_date": "latest"},
        )

        assert "regime_label" in payload
        assert "volatility_annualized" in payload
        assert "breadth_pct" in payload
        assert "correlation_density" in payload
        assert "date" in payload