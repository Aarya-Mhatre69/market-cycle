"""
Market Cycle Specialist Agent Constructor.

Instantiates the specialized Market Cycle Analyst Agent bound to
equity valuation percentiles, ERP spreads, and credit cycle tools.
"""

from typing import Optional, Any
from shankh.utils import load_prompt
from shankh.agents.market.cycle_tools import (
    get_market_cycle_metrics,
    get_liquidity_and_credit_cycle,
    get_cycle_price_structure,
    get_market_cycle_synthesis,
)
from shankh.agents.agent import build_agent


def build_cycle_agent(checkpointer: Optional[Any] = None):
    """
    Constructs the Market Cycle Specialist Agent.

    Args:
        checkpointer: Optional state checkpointer for LangGraph persistence.

    Returns:
        Compiled LangGraph agent runner configured with cycle tools and prompt.
    """
    tools = [
        get_market_cycle_synthesis,
        get_cycle_price_structure,
        get_market_cycle_metrics,
        get_liquidity_and_credit_cycle,
    ]
    system_prompt = load_prompt("market_cycle")
    middlewares = None

    agent = build_agent(
        tools=tools,
        model_name="gpt-4o",
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middlewares=middlewares,
    )
    return agent