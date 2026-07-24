from shankh.agents.shared_tools import get_web_search_tool, filter_tools
from shankh.ml.macro.tool import get_stock_clusters


def get_macro_analyst_tools() -> list:
    """Tools owned by the Macro Analyst only."""
    return filter_tools([
        get_web_search_tool(),
        get_stock_clusters,
    ])
