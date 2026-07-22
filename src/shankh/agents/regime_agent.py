import json
import logging
import os
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from shankh.regime.tools import query_market_regime
from shankh.utils import extract_response_text, resolve_model

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str) -> str:
    """
    Load system prompt markdown from config/prompts/{prompt_name}.md
    """
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


def get_web_search_tool():
    """Safely initialize Tavily web search tool if API key is configured."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        return TavilySearch(
            name="search_web",
            description=(
                "Search the web for current financial news, macro events, regulatory changes, "
                "corporate announcements, or analyst commentary. Input should be a search query."
            ),
            max_results=5,
            search_depth="advanced",
            api_key=tavily_key,
        )
    return None


@tool
def score_market_breadth(advances: int, declines: int, pct_above_20dma: float) -> str:
    """Compute market breadth score."""
    ad_ratio = round(advances / max(declines, 1), 2)
    breadth_score = round(0.5 * min(ad_ratio / 3.0, 1.0) + 0.5 * (pct_above_20dma / 100.0), 3)
    label = "strong" if breadth_score > 0.65 else "moderate" if breadth_score > 0.45 else "weak"
    return json.dumps({
        "ad_ratio": ad_ratio,
        "pct_above_20dma": pct_above_20dma,
        "breadth_score": breadth_score,
        "breadth_label": label
    })


@tool
def classify_volatility_regime(vix: float) -> str:
    """Classify volatility regime from India VIX."""
    if vix < 12:
        label, implication = "complacency", "Extremely low fear"
    elif vix < 17:
        label, implication = "normal", "Healthy risk appetite"
    elif vix < 22:
        label, implication = "elevated", "Caution required"
    else:
        label, implication = "high", "Risk-off behavior likely"
    return json.dumps({"india_vix": vix, "vol_regime": label, "implication": implication})


@tool
def identify_sector_rotation(sector_returns: dict) -> str:
    """Identify market sector rotation dynamics."""
    if not sector_returns:
        return json.dumps({"error": "No sector data provided."})
    sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
    leaders = [s for s, r in sorted_sectors[:3]]
    laggards = [s for s, r in sorted_sectors[-3:]]
    return json.dumps({"leaders": leaders, "laggards": laggards})


def get_regime_analyst_tools() -> list:
    tools = [
        get_web_search_tool(),
        score_market_breadth,
        classify_volatility_regime,
        identify_sector_rotation,
        query_market_regime,
    ]
    return tools


def build_regime_analyst_agent(checkpointer=None):
    """Build the regime analyst sub-agent using prompt from config/prompts/regime_analyst.md"""
    system_prompt = load_prompt("regime_analyst")
    return create_agent(
        model=resolve_model(provider_hint="cerebras"),
        tools=get_regime_analyst_tools(),
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )



def run_regime_analysis(market_snapshot: dict, sector_data: dict, thread_id: str = "regime-default") -> str:
    """Run the regime analyst on pre-fetched market data."""
    agent = build_regime_analyst_agent(checkpointer=MemorySaver())
    prompt = (
        "Here is current market snapshot and sector performance data:\n\n"
        f"Snapshot:\n```json\n{json.dumps(market_snapshot, indent=2)}\n```\n\n"
        f"Sectors:\n```json\n{json.dumps(sector_data, indent=2)}\n```\n\n"
        "You can use web search if you need market context or news. "
        "Classify the regime using the required output structure."
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )
    return extract_response_text(result.get("messages", []))
