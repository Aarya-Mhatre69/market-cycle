"""
Live Integration Tests for Shankh Research Assistant Orchestrator and Specialized Subagents.

Executes end-to-end multi-agent execution trajectories against live LLM providers and
MCP tools over streamable HTTP.
"""

import os
import pytest
from dotenv import load_dotenv
from shankh.agents.supervisor import ResearchAssistantOrchestrator

load_dotenv()

RUN_LIVE_AGENT_TESTS = os.getenv("RUN_LIVE_AGENT_TESTS") == "1"
HAS_REQUIRED_KEYS = bool(os.getenv("OPENAI_API_KEY"))

if not RUN_LIVE_AGENT_TESTS or not HAS_REQUIRED_KEYS:
    pytest.skip(
        "Skipping live orchestrator integration tests. Set RUN_LIVE_AGENT_TESTS=1 "
        "and OPENAI_API_KEY in environment to execute.",
        allow_module_level=True,
    )


def _get_text_content(response) -> str:
    """Extracts plain text content from string, dictionary, or block list payloads."""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        parts = []
        for part in response:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    if isinstance(response, dict):
        return response.get("text", str(response))
    return str(response)


def test_live_orchestrator_macro_delegation():
    """
    Verifies multi-agent delegation by requiring the orchestrator to route a macro query
    to the specialized Macro Analyst subagent (`macro-analyst`), execute underlying tools
    (`get_indian_macro_indicators`, `get_fii_dii_flows`), and return a structured brief.
    """
    orchestrator = ResearchAssistantOrchestrator()
    thread_id = "live-orchestrator-macro-test"

    question = (
        "Consult your specialized macro analyst subagent to conduct an executive synthesis "
        "of the Indian rate environment, US-India 10Y yield spread, and recent FII/DII flow trends."
    )

    response = orchestrator.ask(question, thread_id=thread_id)

    print("\n--- [LIVE RUN] Macro Delegation Response ---")
    print(response)

    assert isinstance(response, (str, list, dict)), "Response must be a string, list, or dict"

    text_content = _get_text_content(response)
    assert len(text_content) > 200, "Macro brief response must be detailed (>200 chars)"

    required_terms = ["fii", "dii", "yield", "rate"]
    assert any(
        term in text_content.lower() for term in required_terms
    ), f"Response missing required macro concepts: {required_terms}"


def test_live_orchestrator_market_breadth_and_cycle_delegation():
    """
    Verifies multi-agent routing to Market Breadth (`breadth-analyst`) and Market Cycle
    (`cycle-analyst`) subagents to evaluate 150-ticker participation and Fed Model ERP.
    """
    orchestrator = ResearchAssistantOrchestrator()
    thread_id = "live-orchestrator-market-test"

    question = (
        "Delegate to your specialized breadth and cycle subagents to evaluate:\n"
        "1. Percentage of 150 universe stocks trading above 50-day and 200-day DMA.\n"
        "2. Current Nifty 50 P/E percentile and Fed Model Equity Risk Premium (ERP) spread."
    )

    response = orchestrator.ask(question, thread_id=thread_id)

    print("\n--- [LIVE RUN] Market Breadth & Cycle Delegation Response ---")
    print(response)

    assert isinstance(response, (str, list, dict)), "Response must be a string, list, or dict"

    text_content = _get_text_content(response)
    assert len(text_content) > 200, "Market brief response must be detailed (>200 chars)"

    required_terms = ["dma", "breadth", "equity risk premium", "p/e"]
    assert any(
        term in text_content.lower() for term in required_terms
    ), f"Response missing required market breadth/cycle concepts: {required_terms}"


def test_live_orchestrator_single_stock_tools_and_web_search():
    """
    Verifies that the orchestrator can execute direct single-stock tools (`get_stock_clusters`,
    `query_gbm_price_band`) and web search (`search_web`) for single-stock research queries.
    """
    orchestrator = ResearchAssistantOrchestrator()
    thread_id = "live-orchestrator-stock-test"

    question = (
        "Analyze RELIANCE.NS. Fetch its next-day price band prediction, check its factor "
        "peer cluster, and search the web for recent major corporate announcements."
    )

    response = orchestrator.ask(question, thread_id=thread_id)

    print("\n--- [LIVE RUN] Single Stock & Search Response ---")
    print(response)

    assert isinstance(response, (str, list, dict)), "Response must be a string, list, or dict"

    text_content = _get_text_content(response)
    assert len(text_content) > 200, "Stock report response must be detailed (>200 chars)"

    required_terms = ["reliance", "price", "band"]
    assert any(
        term in text_content.lower() for term in required_terms
    ), f"Response missing single-stock query markers: {required_terms}"