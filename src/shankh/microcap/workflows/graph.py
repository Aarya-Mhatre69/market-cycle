from typing import Any, TypedDict

from langgraph.graph import StateGraph


class MicrocapState(TypedDict):
    ticker: str
    filings: list[bytes] | None
    extracted_entities: dict[str, Any] | None
    chunks: list[str] | None
    red_flags: list[dict[str, Any]] | None
    llm_explanations: list[dict[str, Any]] | None
    rag_answers: list[dict[str, Any]] | None
    risk_score: dict[str, Any] | None
    report: str | None
    error: str | None


def fetch_filings_node(state: MicrocapState) -> MicrocapState:
    ...


def extract_entities_node(state: MicrocapState) -> MicrocapState:
    ...


def chunk_and_index_node(state: MicrocapState) -> MicrocapState:
    ...


def red_flag_rules_node(state: MicrocapState) -> MicrocapState:
    ...


def llm_explain_node(state: MicrocapState) -> MicrocapState:
    ...


def rag_qa_node(state: MicrocapState) -> MicrocapState:
    ...


def score_and_report_node(state: MicrocapState) -> MicrocapState:
    ...


def build_microcap_graph() -> StateGraph:
    workflow = StateGraph(MicrocapState)
    workflow.add_node("fetch_filings", fetch_filings_node)
    workflow.add_node("extract", extract_entities_node)
    workflow.add_node("chunk_index", chunk_and_index_node)
    workflow.add_node("red_flags", red_flag_rules_node)
    workflow.add_node("llm_explain", llm_explain_node)
    workflow.add_node("rag_qa", rag_qa_node)
    workflow.add_node("score_report", score_and_report_node)
    workflow.set_entry_point("fetch_filings")
    workflow.add_edge("fetch_filings", "extract")
    workflow.add_edge("extract", "chunk_index")
    workflow.add_edge("extract", "red_flags")
    workflow.add_edge("red_flags", "llm_explain")
    workflow.add_edge("chunk_index", "rag_qa")
    workflow.add_edge("llm_explain", "score_report")
    workflow.add_edge("rag_qa", "score_report")
    return workflow.compile()
