from typing import Any, TypedDict

from langgraph.graph import StateGraph


class PriceBandState(TypedDict):
    ticker: str
    raw_data: list[dict[str, Any]]
    features: dict[str, Any]
    model_name: str
    prediction: dict[str, Any]
    evaluation: dict[str, Any]
    error: str | None


def ingest_node(state: PriceBandState) -> PriceBandState:
    ...


def validate_node(state: PriceBandState) -> PriceBandState:
    ...


def feature_engineering_node(state: PriceBandState) -> PriceBandState:
    ...


def model_selector_node(state: PriceBandState) -> PriceBandState:
    ...


def evaluate_node(state: PriceBandState) -> PriceBandState:
    ...


def persist_forecast_node(state: PriceBandState) -> PriceBandState:
    ...


def build_price_band_graph() -> StateGraph:
    workflow = StateGraph(PriceBandState)
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("model_selector", model_selector_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("persist", persist_forecast_node)
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "validate")
    workflow.add_edge("validate", "feature_engineering")
    workflow.add_edge("feature_engineering", "model_selector")
    workflow.add_edge("model_selector", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        lambda s: "persist" if s.get("evaluation", {}).get("accuracy_ok") else "log_failure",
    )
    return workflow.compile()
