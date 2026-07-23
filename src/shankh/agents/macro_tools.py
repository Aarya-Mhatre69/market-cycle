from src.shankh.agents.shared_tools import get_web_search_tool


def get_macro_analyst_tools() -> list:
    """Tools owned by the Macro Analyst only.

    No production macro model exists yet, so this agent gets live search only.
    """
    return [
        get_web_search_tool(),
    ]
