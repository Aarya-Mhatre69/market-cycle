from typing import Any, TypedDict

from langgraph.graph import StateGraph


class SimulationState(TypedDict):
    event_seed: str
    scenarios: list[dict[str, Any]]
    personas: list[dict[str, Any]]
    rounds: list[dict[str, Any]]
    narrative: str | None
    baseline_metric: dict[str, Any] | None
    simulation_metric: dict[str, Any] | None
    error: str | None


def event_seeder_node(state: SimulationState) -> SimulationState:
    ...


def scenario_generator_node(state: SimulationState) -> SimulationState:
    ...


def simulation_rounds_node(state: SimulationState) -> SimulationState:
    ...


def narrative_synthesis_node(state: SimulationState) -> SimulationState:
    ...


def baseline_comparison_node(state: SimulationState) -> SimulationState:
    ...


def build_mirofish_graph() -> StateGraph:
    workflow = StateGraph(SimulationState)
    workflow.add_node("seed", event_seeder_node)
    workflow.add_node("scenarios", scenario_generator_node)
    workflow.add_node("simulate", simulation_rounds_node)
    workflow.add_node("narrative", narrative_synthesis_node)
    workflow.add_node("compare_baseline", baseline_comparison_node)
    workflow.set_entry_point("seed")
    workflow.add_edge("seed", "scenarios")
    workflow.add_edge("scenarios", "simulate")
    workflow.add_edge("simulate", "narrative")
    workflow.add_edge("narrative", "compare_baseline")
    return workflow.compile()
