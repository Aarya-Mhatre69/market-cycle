import json
import logging
import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
from shankh.macro.tools import query_stock_clustering_and_forensics
logger = logging.getLogger(__name__)
from pathlib import Path


def load_prompt(prompt_name: str) -> str:
    """
    Load system prompt markdown from config/prompts/{prompt_name}.md
    """
    # Navigate relative to project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    
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
    query_stock_clustering_and_forensics
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


from shankh.utils import extract_response_text

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