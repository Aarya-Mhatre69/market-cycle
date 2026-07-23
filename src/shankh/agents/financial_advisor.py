import logging

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.shankh.ml.price_band.tool import query_xgboost_price_band
from src.shankh.agents.company_tools import (
    get_company_analyst_tools,
    query_forensic_red_flags,
    query_stock_peers,
)
from src.shankh.agents.macro_tools import get_macro_analyst_tools
from src.shankh.agents.market_tools import get_market_analyst_tools, query_market_regime
from src.shankh.agents.shared_tools import (
    get_web_search_tool,
)

from src.shankh.utils import extract_response_text, resolve_model,load_prompt


load_dotenv()

logger = logging.getLogger(__name__)


def _build_subagents() -> list[dict]:
    """Build specialized subagent definitions for the supervisor."""
    return [
        {
            "name": "macro-analyst",
            "description": (
                "Indian macroeconomic analysis. Delegate RBI policy, repo rate, inflation, "
                "CPI/WPI, bond yields, USDINR, crude oil, FII/DII flows, global markets, "
                "and economic event questions."
            ),
            "system_prompt": load_prompt("macro_analyst"),
            "tools": get_macro_analyst_tools(),
            "model": resolve_model(provider_hint="mistral"),
        },
        {
            "name": "market-analyst",
            "description": (
                "Indian equity market analysis. Delegate market state, volatility, breadth, "
                "sector rotation, correlation, market cycle, and broad market environment questions."
            ),
            "system_prompt": load_prompt("market_analyst"),
            "tools": get_market_analyst_tools(),
            "model": resolve_model(provider_hint="cerebras"),
        },
        {
            "name": "company-analyst",
            "description": (
                "Stock-level analysis. Delegate price forecasting, technical and fundamental context, "
                "peer clustering, forensic screening, company news, and single-company research."
            ),
            "system_prompt": load_prompt("company_analyst"),
            "tools": get_company_analyst_tools(),
            "model": resolve_model(provider_hint="mistral"),
        },
    ]


def get_advisor_tools() -> list:
    """Supervisor tools: real production tools only, with deep reasoning delegated to subagents."""
    return [
        get_web_search_tool(),
        query_market_regime,
        query_xgboost_price_band,
        query_stock_peers,
        query_forensic_red_flags,
    ]


def build_financial_advisor_agent(checkpointer=None):
    """Build the Financial Advisor supervisor agent with specialized subagents."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    return create_deep_agent(
        model=resolve_model(),
        tools=get_advisor_tools(),
        system_prompt=load_prompt("financial_advisor"),
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
