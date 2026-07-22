
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch

from shankh.utils import extract_response_text, extract_text_content
from shankh.mcp.tools.market_data_server import (
    macro_indicators,
    market_regime_signal,
    market_snapshot,
    sector_performance,
    stock_quote,
)
from shankh.agents.macro_agent import run_macro_analysis
from shankh.agents.regime_agent import run_regime_analysis

logger = logging.getLogger(__name__)

def load_prompt(prompt_name: str) -> str:
    """
    Load system prompt markdown from config/prompts/{prompt_name}.md
    """
    # Navigate relative to project root
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

logger = logging.getLogger(__name__)

search_web = TavilySearch(
    name="search_web",
    description=(
        "Search the web for current financial news, macro events, regulatory changes, "
        "corporate announcements, or analyst commentary. Input should be a search query."
    ),
    max_results=5,
    search_depth="advanced",
    api_key=os.environ["TAVILY_API_KEY"]
)

@tool
def interpret_rate_environment(repo_rate: float, gsec_10yr: float, cpi: float) -> str:
    """Interpret real interest rates and yield spread."""
    real_rate = round(repo_rate - cpi, 2)
    yield_spread = round(gsec_10yr - repo_rate, 2)
    stance = "accommodative" if repo_rate < cpi else "neutral" if real_rate < 1.0 else "restrictive"
    curve = "steep" if yield_spread > 1.0 else "flat" if abs(yield_spread) < 0.3 else "normal"

    return json.dumps({
        "real_rate_pct": real_rate,
        "yield_spread_bps": int(yield_spread * 100),
        "monetary_stance": stance,
        "yield_curve": curve,
        "interpretation": f"Real rate of {real_rate:.1f}% signals {stance} conditions. Curve is {curve}."
    })


@tool
def score_fii_dii_flows(fii_30d_cr: float, dii_30d_cr: float) -> str:
    """Score institutional flow dynamics."""
    net = fii_30d_cr + dii_30d_cr
    bias = "positive" if net > 0 else "negative"
    return json.dumps({
        "fii_30d_cr": fii_30d_cr,
        "dii_30d_cr": dii_30d_cr,
        "net_30d_cr": round(net, 0),
        "net_flow_bias": bias,
        "interpretation": f"Net institutional 30-day flow is {bias} ({net:+,.0f} Cr)."
    })


MACRO_ANALYST_TOOLS = [
    search_web,  # Search tool included
    interpret_rate_environment,
    score_fii_dii_flows,
]


def _resolve_model():
    if os.getenv("MISTRALAI_API_KEY"):
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRALAI_API_KEY"))
    if os.getenv("GOOGLE_API_KEY"):
        return "google_genai:gemini-2.0-flash"
    return "google_genai:gemini-2.5-flash"


def build_macro_analyst_agent(checkpointer=None):
    """Build the macro analyst sub-agent using system prompt from config/prompts/macro_analyst.md"""
    system_prompt = load_prompt("macro_analyst")
    return create_agent(
        model=_resolve_model(),
        tools=MACRO_ANALYST_TOOLS,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )


def run_macro_analysis(macro_indicators: dict, thread_id: str = "macro-default") -> str:
    """Run macro analyst on raw indicator data."""
    agent = build_macro_analyst_agent(checkpointer=MemorySaver())
    prompt = (
        "Here are the current macro indicators for the Indian market:\n\n"
        f"```json\n{json.dumps(macro_indicators, indent=2)}\n```\n\n"
        "You can use web search if you need live news context on RBI policy or global macro events. "
        "Provide your macro regime assessment using the exact required output structure."
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )
    return extract_response_text(result.get("messages", []))
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


# Delegation Tools to Sub-Agents
@tool
def consult_macro_analyst(context: str) -> str:
    """Delegate to Shankh Macro Analyst for deep macro regime analysis."""
    try:
        macro_raw = macro_indicators()
        res = run_macro_analysis(
            macro_indicators=macro_raw,
            thread_id=f"macro-{hash(context) % 100000}"
        )
        return extract_text_content(res)
    except Exception as exc:
        logger.warning("Macro analyst error: %s", exc)
        return f"Macro analyst unavailable: {exc}"


@tool
def consult_regime_analyst(context: str) -> str:
    """Delegate to Shankh Regime Analyst for deep market regime analysis."""
    try:
        snapshot_raw = market_snapshot()
        sector_raw = sector_performance(period="1M")
        res = run_regime_analysis(
            market_snapshot=snapshot_raw,
            sector_data=sector_raw,
            thread_id=f"regime-{hash(context) % 100000}"
        )
        return extract_text_content(res)
    except Exception as exc:
        logger.warning("Regime analyst error: %s", exc)
        return f"Regime analyst unavailable: {exc}"

search_web = TavilySearch(
    name="search_web",
    description=(
        "Search the web for current financial news, macro events, regulatory changes, "
        "corporate announcements, or analyst commentary. Input should be a search query."
    ),
    max_results=5,
    search_depth="advanced",
    api_key=os.environ["TAVILY_API_KEY"]
)


# Combined Supervisor Tools
ADVISOR_TOOLS = [
    search_web,
    fetch_market_snapshot,
    fetch_macro_indicators,
    fetch_sector_performance,
    fetch_stock_quote,
    fetch_regime_signal,
    consult_macro_analyst,
    consult_regime_analyst,
]


def _resolve_model() -> str:
    if os.getenv("GOOGLE_API_KEY"):
        return "google_genai:gemini-2.5-flash"
    raise RuntimeError("No configured LLM API key found.")


def build_financial_advisor_graph(checkpointer=None):
    """Build and compile the Financial Advisor supervisor graph."""
    from typing_extensions import TypedDict

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    if checkpointer is None:
        checkpointer = MemorySaver()

    # Dynamic loading of system prompt from config/prompts/financial_advisor.md
    system_prompt = load_prompt("financial_advisor")

    agent = create_agent(
        model=_resolve_model(),
        tools=ADVISOR_TOOLS,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )

    tool_node = ToolNode(ADVISOR_TOOLS, handle_tool_errors=True)

    def advisor_node(state: State) -> dict:
        result = agent.invoke(
            {"messages": state["messages"]},
            config={"recursion_limit": 20},
        )
        return {"messages": result["messages"]}

    def tools_node(state: State) -> dict:
        return tool_node.invoke({"messages": state["messages"]})

    def should_continue(state: State) -> Literal["tools", "__end__"]:
        last = state["messages"][-1] if state["messages"] else None
        if last and getattr(last, "tool_calls", None):
            return "tools"
        return "__end__"

    workflow = StateGraph(State)
    workflow.add_node("advisor", advisor_node, retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0))
    workflow.add_node("tools", tools_node, retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.5))

    workflow.add_edge(START, "advisor")
    workflow.add_conditional_edges("advisor", should_continue, ["tools", "__end__"])
    workflow.add_edge("tools", "advisor")

    return workflow.compile(checkpointer=checkpointer)


class FinancialAdvisor:
    """Interface to Shankh Financial Advisor."""

    def __init__(self, checkpointer=None) -> None:
        self._checkpointer = checkpointer or MemorySaver()
        self._graph = build_financial_advisor_graph(self._checkpointer)

    def ask(self, question: str, thread_id: str = "default", recursion_limit: int = 20) -> str:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
        result = self._graph.invoke({"messages": [HumanMessage(content=question)]}, config)
        return extract_response_text(result.get("messages", []))

