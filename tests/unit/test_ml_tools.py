"""
Unit tests for the MCP-adapter tool wiring in the new decoupled architecture.

The legacy in-process analyst agents (market_agent / macro_agent / company_agent)
were removed. ML tools now run in standalone FastMCP servers and are loaded into
subagents via langchain_mcp_adapters in financial_advisor.py. These tests verify:

  - MCP_SERVER_CONFIG points at the three decoupled ML servers.
  - _partition_mcp_tools routes the fetched remote tools to the correct
    analyst subagent buckets.

Direct wrapper-tool invocation tests (via toolname.invoke()) live in test_tools.py.

Run:
    uv run pytest tests/unit/test_ml_tools.py -v -s
"""

import logging
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@tool
def get_stock_clusters(tickers: str) -> str:
    """Dummy macro MCP tool: cluster + forensic screen."""
    return f"{{'tickers': {tickers!r}}}"


@tool
def get_market_regime(query_date: str = None) -> str:
    """Dummy market MCP tool: HMM regime."""
    return "{'regime': 'defensive'}"


@tool
def query_gbm_price_band(ticker: str, current_price: float = None) -> str:
    """Dummy price-band MCP tool: LightGBM quantile band."""
    return "{'band': 'high' , 'low': 100}"


@tool
def unrelated_tool() -> str:
    """A tool that should NOT be routed to any analyst bucket."""
    return "unrelated"


@tool
def search_web(query: str) -> str:
    """Dummy Tavily web search tool."""
    return f"search results for {query!r}"


def _fake_web_search():
    return search_web


class TestMCPAdapterConfiguration:

    def test_mcp_server_config_has_three_decoupled_servers(self):
        from shankh.agents.financial_advisor import MCP_SERVER_CONFIG

        assert set(MCP_SERVER_CONFIG) == {
            "macro_server", "market_server", "price_band_server"
        }
        for name, cfg in MCP_SERVER_CONFIG.items():
            assert "url" in cfg, f"{name} missing url"
            assert cfg.get("transport") in ("streamable_http", "streamable-http"), \
                f"{name} has invalid transport: {cfg.get('transport')}"

    def test_mcp_server_config_default_urls(self, monkeypatch):
        import shankh.agents.financial_advisor as fa

        monkeypatch.delenv("MACRO_MCP_URL", raising=False)
        monkeypatch.delenv("MARKET_MCP_URL", raising=False)
        monkeypatch.delenv("PRICE_BAND_MCP_URL", raising=False)

        cfg = fa.MCP_SERVER_CONFIG
        assert cfg["macro_server"]["url"] == "http://localhost:8001/mcp"
        assert cfg["market_server"]["url"] == "http://localhost:8002/mcp"
        assert cfg["price_band_server"]["url"] == "http://localhost:8003/mcp"


class TestPartitionMCPTools:

    def test_routes_remote_tools_to_analyst_buckets(self, monkeypatch):
        import shankh.agents.financial_advisor as fa

        monkeypatch.setattr(fa, "get_web_search_tool", _fake_web_search)

        all_mcp_tools = [
            get_stock_clusters,
            get_market_regime,
            query_gbm_price_band,
            unrelated_tool,
        ]
        buckets = fa._partition_mcp_tools(all_mcp_tools)

        macro_names = {t.name for t in buckets["macro"]}
        market_names = {t.name for t in buckets["market"]}
        company_names = {t.name for t in buckets["company"]}
        supervisor_names = {t.name for t in buckets["supervisor"]}

        assert "get_stock_clusters" in macro_names
        assert "search_web" in macro_names

        assert "get_market_regime" in market_names
        assert "search_web" not in market_names

        assert "query_gbm_price_band" in company_names
        assert "get_stock_clusters" in company_names
        assert "search_web" in company_names

        assert "unrelated_tool" not in macro_names
        assert "unrelated_tool" not in market_names
        assert "unrelated_tool" not in company_names
        assert "unrelated_tool" not in supervisor_names

        assert "get_market_regime" in supervisor_names
        assert "query_gbm_price_band" in supervisor_names
        assert "get_stock_clusters" in supervisor_names
        assert "search_web" in supervisor_names

    def test_survives_missing_optional_tools(self, monkeypatch):
        import shankh.agents.financial_advisor as fa

        monkeypatch.setattr(fa, "get_web_search_tool", lambda: None)

        buckets = fa._partition_mcp_tools([get_market_regime])

        assert buckets["macro"] == []
        assert {t.name for t in buckets["market"]} == {"get_market_regime"}
        assert buckets["company"] == []
        assert {t.name for t in buckets["supervisor"]} == {"get_market_regime"}
