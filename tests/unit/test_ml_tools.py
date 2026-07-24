"""
Live integration tests for ML tool invocation via agents.

Each test builds the real agent, sends a prompt that explicitly instructs
it to call the ML model tool on real data, and asserts the response is
non-empty and substantive.

Requirements:
- Valid API keys in .env (MISTRAL_API_KEY or CEREBRAS_API_KEY or GOOGLE_API_KEY)
- Trained model artifacts on disk for market/macro tools
- Internet access for yfinance and web search

Run:
    uv run pytest tests/unit/test_ml_tools.py -v -s
"""

import logging
import os
import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke(agent, prompt: str, thread_id: str) -> str:
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 15},
    )
    messages = result.get("messages", [])
    last = messages[-1] if messages else None
    content = getattr(last, "content", "") if last else ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content).strip()


# ---------------------------------------------------------------------------
# Market agent — get_market_regime
# ---------------------------------------------------------------------------

class TestMarketAgentToolInvocation:

    def test_market_regime_tool_called_with_real_data(self):
        """
        Build the market analyst agent and ask it to call get_market_regime
        on live Nifty data. Asserts the response is non-empty and contains
        market regime language.
        """
        from shankh.agents.market_agent import build_market_analyst_agent

        agent = build_market_analyst_agent(checkpointer=MemorySaver())

        prompt = (
            "Use the get_market_regime tool to fetch the current Indian equity "
            "market regime from live Nifty 50 data. "
            "Report the regime label, volatility, and breadth. "
            "Do not use web search — only the ML model tool."
        )

        response = _invoke(agent, prompt, thread_id="test-market-regime-live")

        logger.info("Market agent response:\n%s", response)
        assert len(response) > 50, f"Response too short: {response!r}"

        lower = response.lower()
        assert any(kw in lower for kw in ["regime", "volatility", "breadth", "market", "risk"]), \
            f"Expected market regime language in response. Got:\n{response}"


# ---------------------------------------------------------------------------
# Macro agent — get_stock_clusters
# ---------------------------------------------------------------------------

class TestMacroAgentToolInvocation:

    def test_stock_cluster_tool_called_with_real_data(self):
        """
        Build the macro analyst agent and ask it to call get_stock_clusters
        on live NSE data. Asserts the response contains cluster or anomaly info.
        """
        from shankh.agents.agent import build_agent, build_default_pool
        from shankh.agents.macro_tools import get_macro_analyst_tools
        from shankh.utils import load_prompt

        agent = build_agent(
            pool=build_default_pool(),
            tools=get_macro_analyst_tools(),
            system_prompt=load_prompt("macro_analyst"),
            checkpointer=MemorySaver(),
        )

        prompt = (
            "Use the get_stock_clusters tool to run live stock clustering on "
            "the default NSE universe. "
            "Report which cluster INFY.NS and TCS.NS belong to, and list any "
            "forensic anomalies found. "
            "Do not use web search — only the ML model tool."
        )

        response = _invoke(agent, prompt, thread_id="test-macro-clusters-live")

        logger.info("Macro agent response:\n%s", response)
        assert len(response) > 50, f"Response too short: {response!r}"

        lower = response.lower()
        assert any(kw in lower for kw in ["cluster", "infy", "tcs", "anomaly", "forensic", "group"]), \
            f"Expected clustering language in response. Got:\n{response}"


# ---------------------------------------------------------------------------
# Company agent — query_gbm_price_band
# ---------------------------------------------------------------------------

class TestCompanyAgentPriceBandToolInvocation:

    def test_price_band_tool_called_for_reliance(self):
        """
        Build the company analyst agent and ask it to call query_gbm_price_band
        for RELIANCE.NS with live yfinance data. Asserts band prices are returned.
        """
        from shankh.agents.company_agent import build_company_analyst_agent

        agent = build_company_analyst_agent(checkpointer=MemorySaver())

        prompt = (
            "Use the query_gbm_price_band tool to get the predicted next-day "
            "high/low price band for RELIANCE.NS using live market data. "
            "Report the predicted high price, low price, and band width. "
            "Do not use web search — only the ML model tool."
        )

        response = _invoke(agent, prompt, thread_id="test-company-priceband-reliance")

        logger.info("Company agent price-band response:\n%s", response)
        assert len(response) > 50, f"Response too short: {response!r}"

        lower = response.lower()
        assert any(kw in lower for kw in ["price", "band", "high", "low", "reliance", "predicted", "inr"]), \
            f"Expected price band language in response. Got:\n{response}"


# ---------------------------------------------------------------------------
# Company agent — query_stock_peers
# ---------------------------------------------------------------------------

class TestCompanyAgentPeersToolInvocation:

    def test_peers_tool_called_for_infy(self):
        """
        Build the company analyst agent and ask it to call query_stock_peers
        for INFY.NS. Asserts peer group or cluster info is returned.
        """
        from shankh.agents.company_agent import build_company_analyst_agent

        agent = build_company_analyst_agent(checkpointer=MemorySaver())

        prompt = (
            "Use the query_stock_peers tool to find the factor cluster and peer group "
            "for INFY.NS. List the peers and report the cluster id. "
            "Do not use web search — only the ML model tool."
        )

        response = _invoke(agent, prompt, thread_id="test-company-peers-infy")

        logger.info("Company agent peers response:\n%s", response)
        assert len(response) > 30, f"Response too short: {response!r}"

        lower = response.lower()
        assert any(kw in lower for kw in ["cluster", "peer", "infy", "group", "factor"]), \
            f"Expected peer/cluster language in response. Got:\n{response}"


# ---------------------------------------------------------------------------
# Company agent — query_forensic_red_flags
# ---------------------------------------------------------------------------

class TestCompanyAgentForensicsToolInvocation:

    def test_forensic_tool_called_for_universe(self):
        """
        Build the company analyst agent and ask it to call query_forensic_red_flags
        for the full universe. Asserts the response mentions anomalies or a clean bill.
        """
        from shankh.agents.company_agent import build_company_analyst_agent

        agent = build_company_analyst_agent(checkpointer=MemorySaver())

        prompt = (
            "Use the query_forensic_red_flags tool to get the full forensic anomaly "
            "screen for the trained stock universe. "
            "Report how many stocks are flagged and list them. "
            "Do not use web search — only the ML model tool."
        )

        response = _invoke(agent, prompt, thread_id="test-company-forensics-universe")

        logger.info("Company agent forensics response:\n%s", response)
        assert len(response) > 30, f"Response too short: {response!r}"

        lower = response.lower()
        assert any(kw in lower for kw in ["anomaly", "flag", "forensic", "screened", "stock", "universe"]), \
            f"Expected forensic language in response. Got:\n{response}"
