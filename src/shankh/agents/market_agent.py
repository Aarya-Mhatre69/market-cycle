import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from shankh.agents.agent import build_agent,build_default_pool

from shankh.agents.market_tools import (
    get_market_analyst_tools,
)
from shankh.utils import extract_response_text, resolve_model,load_prompt


logger = logging.getLogger(__name__)


def build_market_analyst_agent(checkpointer=None):
    """Build the market analyst agent using config/prompts/market_analyst.md."""
    return build_agent(
        pool=build_default_pool(),
        tools=get_market_analyst_tools(),
        system_prompt=load_prompt("market_analyst"),
        checkpointer=checkpointer,
    )


def run_market_analysis(market_snapshot: dict, sector_data: dict, thread_id: str = "market-default") -> str:
    """Run the market analyst on pre-fetched market data."""
    agent = build_market_analyst_agent(checkpointer=MemorySaver())
    prompt = (
        "Here is current market snapshot and sector performance data:\n\n"
        f"Snapshot:\n```json\n{json.dumps(market_snapshot, indent=2)}\n```\n\n"
        f"Sectors:\n```json\n{json.dumps(sector_data, indent=2)}\n```\n\n"
        "You can use web search if you need market context or news. "
        "Classify the market environment using the required output structure."
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )
    logger.info("market agent called")
    return extract_response_text(result.get("messages", []))
