from typing import Any, TypedDict

from langgraph.graph import StateGraph


class RegimeState(TypedDict):
    date: str
    macro_data: dict[str, Any]
    breadth_data: dict[str, Any]
    market_data: dict[str, Any]
    macro_score: float | None
    breadth_score: float | None
    cycle_label: str | None
    final_regime: dict[str, Any] | None
    error: str | None


def fetch_macro_node(state: RegimeState) -> RegimeState:
    ...


def fetch_breadth_node(state: RegimeState) -> RegimeState:
    ...


def fetch_market_node(state: RegimeState) -> RegimeState:
    ...


def macro_analyst_node(state: RegimeState) -> RegimeState:
    ...


def breadth_agent_node(state: RegimeState) -> RegimeState:
    ...


def cycle_agent_node(state: RegimeState) -> RegimeState:
    ...


def regime_fuser_node(state: RegimeState) -> RegimeState:
    ...


def persist_regime_node(state: RegimeState) -> RegimeState:
    ...


def emit_event_node(state: RegimeState) -> RegimeState:
    ...


def build_regime_graph() -> StateGraph:
    workflow = StateGraph(RegimeState)
    workflow.add_node("fetch_macro", fetch_macro_node)
    workflow.add_node("fetch_breadth", fetch_breadth_node)
    workflow.add_node("fetch_market", fetch_market_node)
    workflow.add_node("macro_analyst", macro_analyst_node)
    workflow.add_node("breadth_agent", breadth_agent_node)
    workflow.add_node("cycle_agent", cycle_agent_node)
    workflow.add_node("fuser", regime_fuser_node)
    workflow.add_node("persist", persist_regime_node)
    workflow.add_node("emit", emit_event_node)
    workflow.set_entry_point("fetch_macro")
    workflow.add_edge("fetch_macro", "macro_analyst")
    workflow.add_edge("fetch_breadth", "breadth_agent")
    workflow.add_edge("fetch_market", "cycle_agent")
    workflow.add_edge("macro_analyst", "fuser")
    workflow.add_edge("breadth_agent", "fuser")
    workflow.add_edge("cycle_agent", "fuser")
    workflow.add_edge("fuser", "persist")
    workflow.add_edge("persist", "emit")
    return workflow.compile()
