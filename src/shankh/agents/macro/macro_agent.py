"""
Macro Analyst Agent Constructor.

Instantiates the domain-pure Macro Analyst Agent bound to macro tools
for institutional flows, yield differentials, forex/commodity momentum,
and global/domestic economic calendar events.
"""

from typing import Optional, Any
from shankh.utils import load_prompt
from shankh.agents.macro.macro_tools import (
    get_fii_dii_flows,
    get_indian_macro_indicators,
    get_forex_and_commodities,
    get_economic_calendar,
)
from shankh.agents.shared_tools import get_web_search_tool
from shankh.agents.agent import build_agent


def build_macro_agent(checkpointer: Optional[Any] = None):
    """
    Constructs the Macro Analyst Agent.

    Args:
        checkpointer: Optional state checkpointer for LangGraph agent persistence.

    Returns:
        Compiled LangGraph agent runner configured with macro tools and system prompt.
    """
    tools = [
        get_fii_dii_flows,
        get_indian_macro_indicators,
        get_forex_and_commodities,
        get_economic_calendar,
        get_web_search_tool()
    ]
    system_prompt = load_prompt("macro_analyst")
    middlewares = None

    agent = build_agent(
        tools=tools,
        model_name="gpt-4o",
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middlewares=middlewares,
    )
    return agent