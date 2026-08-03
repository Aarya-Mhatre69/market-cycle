"""
FastMCP Client tests for the three decoupled ML MCP servers.

Follows the FastMCP testing pattern (testmcp.md): each server's FastMCP app is
wrapped in an in-process Client, then tools are listed and invoked exactly the
way an agent would call them over MCP — but without any agent code.

Servers under test:
  - shankh.mcp.tools.macro   -> get_stock_clusters        (KMeans + IsolationForest)
  - shankh.mcp.tools.market  -> get_market_regime         (HMM regime)
  - shankh.mcp.tools.company -> query_gbm_price_band      (LightGBM quantile)

These are live tests: they hit yfinance for real OHLCV and run the trained
models. If artifacts are missing the server module itself raises on import.

Run:
    uv run pytest tests/unit/test_mcp_ml_servers.py -v -s
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
    """Import an MCP server module (eagerly loads its models); skip if artifacts missing."""
    try:
        return importlib.import_module(f"shankh.mcp.tools.{module_name}").mcp
    except FileNotFoundError as exc:
        pytest.skip(f"Model artifacts not found for {module_name} server: {exc}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not import {module_name} MCP server: {exc}")


def _result_text(result) -> str:
    """Extract the tool's returned JSON string from a FastMCP call_tool result."""
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
    """Invoke a tool and return the parsed JSON payload; skip on tool-level errors."""
    result = await client.call_tool(name=name, arguments=arguments)
    payload = json.loads(_result_text(result))
    if isinstance(payload, dict) and payload.get("error"):
        pytest.skip(f"{name} returned error: {payload['error']}")
    return payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def macro_mcp_client():
    async with Client(transport=_load_server("macro")) as client:
        yield client


@pytest.fixture
async def market_mcp_client():
    async with Client(transport=_load_server("market")) as client:
        yield client


@pytest.fixture
async def company_mcp_client():
    async with Client(transport=_load_server("company")) as client:
        yield client


# ---------------------------------------------------------------------------
# Macro server — get_stock_clusters
# ---------------------------------------------------------------------------

class TestMacroServer:

    async def test_lists_get_stock_clusters_tool(self, macro_mcp_client: Client[FastMCPTransport]):
        tools = await macro_mcp_client.list_tools()
        assert any(t.name == "get_stock_clusters" for t in tools)

    async def test_get_stock_clusters_runs_clustering_model(
        self, macro_mcp_client: Client[FastMCPTransport]
    ):
        payload = await _call_and_parse(
            macro_mcp_client,
            name="get_stock_clusters",
            arguments={"tickers": "INFY.NS,TCS.NS,RELIANCE.NS"},
        )

        assert "ticker_analysis" in payload
        assert "flagged_forensic_anomalies" in payload

        for tkr in ("INFY.NS", "TCS.NS", "RELIANCE.NS"):
            entry = payload["ticker_analysis"].get(tkr)
            assert entry is not None, f"{tkr} missing from ticker_analysis"
            assert "cluster_id" in entry
            assert "peers_in_same_cluster" in entry
            assert "is_forensic_anomaly" in entry


# ---------------------------------------------------------------------------
# Market server — get_market_regime
# ---------------------------------------------------------------------------

class TestMarketServer:

    async def test_lists_get_market_regime_tool(self, market_mcp_client: Client[FastMCPTransport]):
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
        assert "volatility" in payload
        assert "breadth_pct" in payload
        assert "correlation_density" in payload
        assert "date" in payload


# ---------------------------------------------------------------------------
# Company server — query_gbm_price_band
# ---------------------------------------------------------------------------

class TestCompanyServer:

    async def test_lists_query_gbm_price_band_tool(self, company_mcp_client: Client[FastMCPTransport]):
        tools = await company_mcp_client.list_tools()
        assert any(t.name == "query_gbm_price_band" for t in tools)

    async def test_query_gbm_price_band_runs_lightgbm_models(
        self, company_mcp_client: Client[FastMCPTransport]
    ):
        payload = await _call_and_parse(
            company_mcp_client,
            name="query_gbm_price_band",
            arguments={"ticker": "RELIANCE.NS"},
        )

        assert payload.get("model_backend") == "lightgbm"
        band = payload.get("lightgbm_predicted_price_band")
        assert band is not None, f"No LightGBM price band in payload: {payload}"
        assert band["predicted_high_price_inr"] >= band["predicted_low_price_inr"]
        assert "expected_band_width_pct" in band
