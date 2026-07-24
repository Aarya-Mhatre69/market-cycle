from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import logging
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain_cerebras import ChatCerebras
from langgraph.checkpoint.memory import MemorySaver

from shankh.ml.price_band.tool import query_gbm_price_band, query_gbm_price_band
from shankh.agents.company_tools import (
    get_company_analyst_tools,
    query_forensic_red_flags,
    query_stock_peers,
)
from shankh.agents.macro_tools import get_macro_analyst_tools
from shankh.agents.market_tools import get_market_analyst_tools, query_market_regime
from shankh.agents.shared_tools import (
    get_web_search_tool,
    filter_tools,
)

from shankh.utils import extract_response_text, resolve_model, load_prompt


load_dotenv()

logger = logging.getLogger(__name__)


def _build_subagents() -> list[dict]:
    """Build specialized subagent definitions for the supervisor."""
    return [
        {
            "name": "macro-analyst",
            "description": (
                "Indian macroeconomic analysis. Delegate RBI policy, repo rate, inflation, "
                "CPI/WPI, bond yields, USDINR, crude oil, FII/DII flows, global markets, "
                "and economic event questions."
            ),
            "system_prompt": load_prompt("macro_analyst"),
            "tools": get_macro_analyst_tools(),
            "model": resolve_model(provider_hint="mistral"),
        },
        {
            "name": "market-analyst",
            "description": (
                "Indian equity market analysis. Delegate market state, volatility, breadth, "
                "sector rotation, correlation, market cycle, and broad market environment questions."
            ),
            "system_prompt": load_prompt("market_analyst"),
            "tools": get_market_analyst_tools(),
            "model": resolve_model(provider_hint="cerebras"),
        },
        {
            "name": "company-analyst",
            "description": (
                "Stock-level analysis. Delegate price forecasting, technical and fundamental context, "
                "peer clustering, forensic screening, company news, and single-company research."
            ),
            "system_prompt": load_prompt("company_analyst"),
            "tools": get_company_analyst_tools(),
            "model": resolve_model(provider_hint="mistral"),
        },
    ]


def get_advisor_tools() -> list:
    """Supervisor tools: real production tools only, with deep reasoning delegated to subagents."""
    return filter_tools([
        get_web_search_tool(),
        query_market_regime,
        # query_gbm_price_band,
        query_gbm_price_band,
        query_stock_peers,
        query_forensic_red_flags,
    ])


def build_financial_advisor_agent(checkpointer=None):
    """Build the Financial Advisor supervisor agent with specialized subagents."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    return create_deep_agent(
        model=resolve_model("mistral"),
        tools=get_advisor_tools(),
        system_prompt=load_prompt("financial_advisor"),
        subagents=_build_subagents(),
        checkpointer=checkpointer,
    )


class FinancialAdvisor:
    """Interface to Shankh Financial Advisor."""

    def __init__(self, checkpointer=None) -> None:
        self._checkpointer = checkpointer or MemorySaver()
        self._agent = build_financial_advisor_agent(self._checkpointer)

    def ask(self, question: str, thread_id: str = "default", recursion_limit: int = 20) -> str:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
        result = self._agent.invoke({"messages": [HumanMessage(content=question)]}, config)
        messages = result.get("messages", [])
        logger.info("agent returned %d messages", len(messages))
        for i, m in enumerate(messages):
            mtype = type(m).__name__
            content = getattr(m, "content", "")
            tool_calls = getattr(m, "tool_calls", None)
            additional_kwargs = getattr(m, "additional_kwargs", {})
            response_metadata = getattr(m, "response_metadata", {})
            logger.info(
                "  [%d] %s | content=%r | tool_calls=%r | additional_kwargs=%r | response_metadata=%r",
                i, mtype,
                str(content)[:200],
                tool_calls,
                str(additional_kwargs)[:200],
                str(response_metadata)[:200],
            )
        return extract_response_text(messages)


# ---------------------------------------------------------------------------
# New: postgres-backed agent with direct model instances (Mistral supervisor,
# Cerebras subagents). Old functions above are kept as fallback.
# ---------------------------------------------------------------------------

def _get_postgres_checkpointer():
    """Return a PostgresSaver connected to DATABASE_URL, or None if unavailable."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL not set — falling back to MemorySaver")
        return None
    try:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        conn = psycopg.Connection.connect(db_url, autocommit=True)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        return checkpointer
    except Exception as exc:
        logger.warning("Postgres checkpointer unavailable (%s) — falling back to MemorySaver", exc)
        return None


def _build_subagents_v2() -> list[dict]:
    """Subagents using direct Cerebras model instances."""
    # cerebras_model = ChatCerebras(
    #     model="gpt-oss-120b",
    #     api_key=os.getenv("CEREBRAS_API_KEY"),
    # )
    cerebras_model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

    return [
        {
            "name": "macro-analyst",
            "description": (
                "Indian macroeconomic analysis. Delegate RBI policy, repo rate, inflation, "
                "CPI/WPI, bond yields, USDINR, crude oil, FII/DII flows, global markets, "
                "and economic event questions."
            ),
            "system_prompt": load_prompt("macro_analyst"),
            "tools": get_macro_analyst_tools(),
            "model": cerebras_model,
        },
        {
            "name": "market-analyst",
            "description": (
                "Indian equity market analysis. Delegate market state, volatility, breadth, "
                "sector rotation, correlation, market cycle, and broad market environment questions."
            ),
            "system_prompt": load_prompt("market_analyst"),
            "tools": get_market_analyst_tools(),
            "model": cerebras_model,
        },
        {
            "name": "company-analyst",
            "description": (
                "Stock-level analysis. Delegate price forecasting, technical and fundamental context, "
                "peer clustering, forensic screening, company news, and single-company research."
            ),
            "system_prompt": load_prompt("company_analyst"),
            "tools": get_company_analyst_tools(),
            "model": cerebras_model,
        },
    ]


def build_financial_advisor_agent_v2(checkpointer=None):
    """Build the advisor using Mistral as supervisor, Cerebras as subagents, Postgres checkpointer."""
    if checkpointer is None:
        checkpointer = _get_postgres_checkpointer() or MemorySaver()

    mistral_model = ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    return create_deep_agent(
        model=mistral_model,
        tools=get_advisor_tools(),
        system_prompt=load_prompt("financial_advisor"),
        subagents=_build_subagents_v2(),
        checkpointer=checkpointer,
    )


class FinancialAdvisorV2:
    """
    Production Financial Advisor.
    - Mistral Large as supervisor
    - Cerebras (llama-4-scout) as all subagents
    - PostgreSQL checkpointer (falls back to MemorySaver if DB unavailable)
    """

    def __init__(self) -> None:
        self._checkpointer = _get_postgres_checkpointer() or MemorySaver()
        self._agent = build_financial_advisor_agent_v2(self._checkpointer)

    def ask(self, question: str, thread_id: str = "default", recursion_limit: int = 25) -> str:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
        result = self._agent.invoke({"messages": [HumanMessage(content=question)]}, config)
        messages = result.get("messages", [])
        logger.info("v2 agent returned %d messages", len(messages))
        for i, m in enumerate(messages):
            logger.info(
                "  [%d] %s | content=%r | tool_calls=%s",
                i, type(m).__name__,
                str(getattr(m, "content", ""))[:200],
                bool(getattr(m, "tool_calls", None)),
            )
        return extract_response_text(messages)
