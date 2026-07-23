import json
import logging

from src.shankh.agents.agent import build_agent,build_default_pool

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.shankh.agents.macro_tools import (
    get_macro_analyst_tools,
)
from src.shankh.utils import extract_response_text, resolve_model,load_prompt

logger = logging.getLogger(__name__)


def build_macro_analyst_agent(checkpointer=None):
    """Build the macro analyst agent using config/prompts/macro_analyst.md."""
    return build_agent(
        tools=get_macro_analyst_tools(),
        pool=build_default_pool(),
        system_prompt=load_prompt("macro_analyst"),
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
