import json
import logging

from src.shankh.agents.agent import build_agent,build_default_pool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.shankh.agents.company_tools import (
    get_company_analyst_tools,
)
from src.shankh.utils import extract_response_text, resolve_model,load_prompt

logger = logging.getLogger(__name__)


def build_company_analyst_agent(checkpointer=None):
    """Build the company analyst agent using config/prompts/company_analyst.md."""
    return build_agent(
        # model=resolve_model(provider_hint="mistral"),
        pool=build_default_pool(),
        tools=get_company_analyst_tools(),
        system_prompt=load_prompt("company_analyst"),
        checkpointer=checkpointer,
    )


def run_company_analysis(symbol: str, context: dict | None = None, thread_id: str = "company-default") -> str:
    """Run company analyst for a single stock symbol with optional pre-fetched context."""
    agent = build_company_analyst_agent(checkpointer=MemorySaver())
    payload = {"symbol": symbol, "context": context or {}}
    prompt = (
        "Analyze this company using stock-level tools only. Include price-band model output when useful, "
        "peer context, forensic red flags, and relevant company news.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )
    return extract_response_text(result.get("messages", []))
