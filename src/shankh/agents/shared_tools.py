import os

from langchain_tavily import TavilySearch

def get_web_search_tool() -> Optional[TavilySearch]:
    """Safely initialize Tavily web search tool if API key is configured."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return None

    return TavilySearch(
        name="search_web",
        description=(
            "Search the web for real-time qualitative macroeconomic news, RBI Monetary Policy Committee (MPC) commentary, "
            "US Federal Reserve (FOMC) statements, geopolitical energy developments, and government policy updates. "
            "Use this tool ONLY for qualitative context and macro narratives, NOT for basic numerical metrics. "
            "Input must be a concise macro query (e.g., 'RBI MPC rate decision commentary', 'OPEC crude supply news'). "
            "Do NOT search for single-stock equity news or individual company earnings."
        ),
        max_results=5,
        search_depth="advanced",
        api_key=tavily_key,
    )


def filter_tools(tools: list) -> list:
    """Remove None entries from a tools list (e.g. when optional tools are unconfigured)."""
    return [t for t in tools if t is not None]

