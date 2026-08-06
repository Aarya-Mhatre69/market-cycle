"""
Market Breadth Specialist Agent Constructor.

Instantiates the specialized Market Breadth Analyst Agent bound to
150-ticker cross-sectional breadth tools and sector performance matrix tools.
"""

from typing import Optional, Any
from shankh.utils import load_prompt
from shankh.agents.market.breadth_tools import (
    get_market_breadth_metrics,
    get_sector_participation_matrix,
)
from shankh.agents.agent import build_agent


def build_breadth_agent(checkpointer: Optional[Any] = None):
    """
    Constructs the Market Breadth Specialist Agent.

    Args:
        checkpointer: Optional state checkpointer for LangGraph persistence.

    Returns:
        Compiled LangGraph agent runner configured with breadth tools and prompt.
    """
    tools = [
        get_market_breadth_metrics,
        get_sector_participation_matrix,
    ]
    system_prompt = load_prompt("market_breadth")
    middlewares = None

    agent = build_agent(
        tools=tools,
        model_name="gpt-4o",
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middlewares=middlewares,
    )
    return agent