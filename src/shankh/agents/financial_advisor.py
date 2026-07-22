import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from shankh.price_band.tool import query_xgboost_price_band
from shankh.macro.tools import query_stock_clustering_and_forensics
from shankh.regime.tools import query_market_regime
from shankh.mcp.tools.market_data_server import (
    macro_indicators,
    market_regime_signal,
    market_snapshot,
    sector_performance,
    stock_quote,
)
from shankh.utils import extract_response_text, resolve_model

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str) -> str:
    """
    Load system prompt markdown from config/prompts/{prompt_name}.md
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]

    possible_paths = [
        project_root / "config" / "prompts" / f"{prompt_name}.md",
        project_root / "config" / "prompts" / f"{prompt_name}.txt",
        Path("config/prompts") / f"{prompt_name}.md",
    ]

    for path in possible_paths:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        f"Prompt file '{prompt_name}' not found. Searched in: {[str(p) for p in possible_paths]}"
    )


def get_web_search_tool() -> Any:
    """Safely initialize Tavily web search tool if API key is configured."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        return TavilySearch(
            name="search_web",
            description=(
                "Search the web for current financial news, macro events, regulatory changes, "
                "corporate announcements, or analyst commentary. Input should be a search query."
            ),
            max_results=5,
            search_depth="advanced",
            api_key=tavily_key,
        )
    return None


# MCP Market Data Tools
@tool
def fetch_market_snapshot() -> str:
    """Fetch market snapshot (indices, VIX, A/D ratio)."""
    return json.dumps(market_snapshot(), indent=2)


@tool
def fetch_macro_indicators() -> str:
    """Fetch macroeconomic indicators."""
    return json.dumps(macro_indicators(), indent=2)


@tool
def fetch_sector_performance(period: str = "1M") -> str:
    """Fetch sector returns and relative metrics."""
    return json.dumps(sector_performance(period), indent=2)


@tool
def fetch_stock_quote(symbol: str) -> str:
    """Fetch real-time stock quote and metrics."""
    return json.dumps(stock_quote(symbol), indent=2)


@tool
def fetch_regime_signal() -> str:
    """Fetch high-level market regime classification."""
    return json.dumps(market_regime_signal(), indent=2)


# Sub-Agent Definitions (true subagents, not tool wrappers)
def _get_macro_subagent_tools() -> list:
    from shankh.agents.macro_agent import (
        interpret_rate_environment,
        score_fii_dii_flows,
        get_web_search_tool,
    )
    tools = [
        get_web_search_tool(),
        interpret_rate_environment,
        score_fii_dii_flows,
        query_stock_clustering_and_forensics,
    ]
    return tools


def _get_regime_subagent_tools() -> list:
    from shankh.agents.regime_agent import (
        score_market_breadth,
        classify_volatility_regime,
        identify_sector_rotation,
        get_web_search_tool,
    )
    tools = [
        get_web_search_tool(),
        score_market_breadth,
        classify_volatility_regime,
        identify_sector_rotation,
        query_market_regime,
    ]
    return tools


def _build_subagents() -> list:
    from shankh.agents.macro_agent import load_prompt as load_macro_prompt
    from shankh.agents.regime_agent import load_prompt as load_regime_prompt

    return [
        {
            "name": "macro-analyst",
            "description": (
                "Deep macro regime analysis for the Indian market. "
                "Delegate macroeconomic questions, RBI policy analysis, "
                "FII/DII flow assessment, and rate environment interpretation."
            ),
            "system_prompt": load_macro_prompt("macro_analyst"),
            "tools": _get_macro_subagent_tools(),
            "model": resolve_model(provider_hint="mistral"),
        },
        {
            "name": "regime-analyst",
            "description": (
                "Market regime classification and sector rotation analysis. "
                "Delegate market regime detection, volatility regime assessment, "
                "breadth analysis, and sector rotation questions."
            ),
            "system_prompt": load_regime_prompt("regime_analyst"),
            "tools": _get_regime_subagent_tools(),
            "model": resolve_model(provider_hint="cerebras"),
        },
    ]


def get_advisor_tools() -> list:
    tools = [
        get_web_search_tool(),
        fetch_market_snapshot,
        fetch_macro_indicators,
        fetch_sector_performance,
        fetch_stock_quote,
        fetch_regime_signal,
        query_xgboost_price_band,
    ]
    return tools


def build_financial_advisor_agent(checkpointer=None):
    """Build the Financial Advisor supervisor agent with true subagents."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    system_prompt = load_prompt("financial_advisor")

    return create_deep_agent(
        model=resolve_model(),
        tools=get_advisor_tools(),
        system_prompt=system_prompt,
        subagents=_build_subagents(),
        checkpointer=checkpointer,
    )


class FinancialAdvisor:
    """Interface to Shankh Financial Advisor."""

    def __init__(self, checkpointer=None) -> None:
        self._checkpointer = checkpointer or MemorySaver()
        self._agent = build_financial_advisor_agent(self._checkpointer)

    def ask(self, question: str, thread_id: str = "default", recursion_limit: int = 20) -> str:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
        result = self._agent.invoke({"messages": [HumanMessage(content=question)]}, config)
        return extract_response_text(result.get("messages", []))
