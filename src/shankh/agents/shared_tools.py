import os
from typing import Any

from langchain_tavily import TavilySearch


def get_web_search_tool() -> Any | None:
    """Safely initialize Tavily web search tool if API key is configured."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return None

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



