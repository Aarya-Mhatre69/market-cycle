"""
Research Assistant Orchestrator Module.

Instantiates the top-level supervisor orchestrator using deepagents and LangGraph.
Directly binds specialized compiled subagents created via build_*_agent factory functions:
- Macro Analyst Agent (macro_agent.py)
- Market Breadth Analyst Agent (breadth_agent.py)
- Market Cycle Analyst Agent (cycle_agent.py)
- Equity Analyst Agent (equity_agent.py)
"""

import asyncio
import logging
import os
from typing import Optional, Any, List

import psycopg
from dotenv import load_dotenv
from deepagents import CompiledSubAgent, create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from shankh.utils import extract_response_text, load_prompt
from shankh.agents.macro.macro_agent import build_macro_agent
from shankh.agents.market.breadth_agent import build_breadth_agent
from shankh.agents.market.cycle_agent import build_cycle_agent
from shankh.agents.shared_tools import get_web_search_tool

load_dotenv()
logger = logging.getLogger(__name__)


def _get_postgres_checkpointer():
    """Return a PostgresSaver connected to DATABASE_URL or None if unconfigured."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL not set. Running orchestrator without persistent checkpointer.")
        return None

    try:
        conn = psycopg.Connection.connect(db_url, autocommit=True)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        return checkpointer
    except Exception as exc:
        logger.warning("Failed to connect to Postgres checkpointer: %s. Continuing in-memory.", exc)
        return None


def build_research_assistant_orchestrator(
    checkpointer: Optional[Any] = None,
    model_name: str = "openai:gpt-4o",
):
    """
    Builds the top-level Research Assistant Orchestrator by directly binding
    specialized compiled subagents constructed via build_*_agent factory functions.
    """
    if checkpointer is None:
        checkpointer = _get_postgres_checkpointer()

    macro_subagent = CompiledSubAgent(
        name="macro-analyst",
        description=(
            "Indian macroeconomic analysis. Delegate RBI policy stance, yield spreads (IN10Y - US10Y), "
            "CPI inflation, FII/DII cash flows, Brent Crude velocity, USD/INR trends, and economic calendar events."
        ),
        runnable=build_macro_agent(),
    )

    breadth_subagent = CompiledSubAgent(
        name="breadth-analyst",
        description=(
            "Cross-sectional market breadth analysis across 150+ tickers. Delegate % of stocks above "
            "20/50/200 DMA, Net 52-Week Highs/Lows, McClellan Oscillator, and sector participation matrix."
        ),
        runnable=build_breadth_agent(),
    )

    cycle_subagent = CompiledSubAgent(
        name="cycle-analyst",
        description=(
            "Market cycle and macro valuation positioning. Delegate Nifty P/E percentiles, "
            "Fed Model Equity Risk Premium (ERP) spread, bank credit growth, and monetary cycle phase."
        ),
        runnable=build_cycle_agent(),
    )

    # equity_subagent = CompiledSubAgent(
    #     name="equity-analyst",
    #     description=(
    #         "Single-stock equity analysis. Delegate LightGBM price band forecasting, "
    #         "peer fundamental clustering, and small-cap stock discovery."
    #     ),
    #     runnable=build_equity_agent(),
    # )

    subagents: List[CompiledSubAgent] = [
        macro_subagent,
        breadth_subagent,
        cycle_subagent,
        # equity_subagent,
    ]

    web_search = get_web_search_tool()
    supervisor_tools = [web_search] if web_search else []

    system_prompt = load_prompt("research_assistant")

    return create_deep_agent(
        model=model_name,
        tools=supervisor_tools,
        system_prompt=system_prompt,
        subagents=subagents,
        checkpointer=checkpointer,
    )


class ResearchAssistantOrchestrator:
    """
    Top-Level Supervisor Research Assistant Interface.
    Orchestrates specialized subagents for macro, breadth, cycle, and equity research.
    """

    def __init__(
        self,
        checkpointer: Optional[Any] = None,
        model_name: str = "openai:gpt-4o",
    ) -> None:
        self._checkpointer = checkpointer or _get_postgres_checkpointer()
        self._model_name = model_name
        self._agent = build_research_assistant_orchestrator(
            checkpointer=self._checkpointer,
            model_name=self._model_name,
        )

    async def ask_async(
        self, question: str, thread_id: str = "default", recursion_limit: int = 50
    ) -> str:
        """Asynchronously invokes the orchestrator agent and returns response text."""
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
        }

        result = await self._agent.ainvoke(
            {"messages": [HumanMessage(content=question)]}, config
        )
        messages = result.get("messages", [])

        logger.info("Orchestrator agent returned %d trajectory messages", len(messages))
        return extract_response_text(messages)

    def ask(
        self, question: str, thread_id: str = "default", recursion_limit: int = 25
    ) -> str:
        """Synchronous wrapper for ask_async."""
        return asyncio.run(
            self.ask_async(
                question=question,
                thread_id=thread_id,
                recursion_limit=recursion_limit,
            )
        )