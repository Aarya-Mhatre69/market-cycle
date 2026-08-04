import asyncio
import logging
import os
from typing import Dict 
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from shankh.agents.shared_tools import filter_tools, get_web_search_tool
from shankh.utils import extract_response_text, load_prompt

load_dotenv()

logger = logging.getLogger(__name__)

# Configurable MCP endpoints for decoupled ML processes
MCP_SERVER_CONFIG = {
    "macro_server": {
        "url": os.getenv("MACRO_MCP_URL", "http://localhost:8001/mcp"),
        "transport": "streamable_http",
    },
    "market_server": {
        "url": os.getenv("MARKET_MCP_URL", "http://localhost:8002/mcp"),
        "transport": "streamable_http",
    },
    "price_band_server": {
        "url": os.getenv("PRICE_BAND_MCP_URL", "http://localhost:8003/mcp"),
        "transport": "streamable_http",
    },
}


def _get_postgres_checkpointer():
    """Return a PostgresSaver connected to DATABASE_URL, or MemorySaver if unavailable."""
    db_url = os.getenv("DATABASE_URL")

    conn = psycopg.Connection.connect(db_url, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer


def _partition_mcp_tools(all_mcp_tools: list) -> Dict[str, list]:
    """
    Organizes tools retrieved dynamically via MCP into analyst subagent buckets.
    """
    tool_map = {tool.name: tool for tool in all_mcp_tools}
    web_search = get_web_search_tool()

    # Macro analyst tools (MCP Macro + Web Search)
    macro_tools = filter_tools([web_search])
    if "get_stock_clusters" in tool_map:
        macro_tools.append(tool_map["get_stock_clusters"])

    # Market analyst tools (MCP Market Regime)
    market_tools = []
    if "get_market_regime" in tool_map:
        market_tools.append(tool_map["get_market_regime"])

    # Company analyst tools (MCP Price Band + Stock Clusters + Web Search)
    company_tools = filter_tools([web_search])
    for tool_name in ["query_gbm_price_band", "get_stock_clusters"]:
        if tool_name in tool_map:
            company_tools.append(tool_map[tool_name])

    # Supervisor tools (High-level entry points for quick routing)
    supervisor_tools = filter_tools([
        web_search,
        tool_map.get("get_market_regime"),
        tool_map.get("query_gbm_price_band"),
        tool_map.get("get_stock_clusters"),
    ])

    return {
        "macro": macro_tools,
        "market": market_tools,
        "company": company_tools,
        "supervisor": supervisor_tools,
    }


def _build_subagents(llm: ChatOpenAI, tools_by_role: Dict[str, list]) -> list[dict]:
    """Build specialized subagent definitions using dynamic MCP tool definitions."""
    return [
        {
            "name": "macro-analyst",
            "description": (
                "Indian macroeconomic analysis. Delegate RBI policy, repo rate, inflation, "
                "CPI/WPI, bond yields, USDINR, crude oil, FII/DII flows, global markets, "
                "and economic event questions."
            ),
            "system_prompt": load_prompt("macro_analyst"),
            "tools": tools_by_role["macro"],
            "model": llm,
        },
        {
            "name": "market-analyst",
            "description": (
                "Indian equity market analysis. Delegate market state, volatility, breadth, "
                "sector rotation, correlation, market cycle, and broad market environment questions."
            ),
            "system_prompt": load_prompt("market_analyst"),
            "tools": tools_by_role["market"],
            "model": llm,
        },
        {
            "name": "company-analyst",
            "description": (
                "Stock-level analysis. Delegate price forecasting, technical and fundamental context, "
                "peer clustering, forensic screening, company news, and single-company research."
            ),
            "system_prompt": load_prompt("company_analyst"),
            "tools": tools_by_role["company"],
            "model": llm,
        },
    ]


async def build_financial_advisor_agent_async(
    client: MultiServerMCPClient,
    checkpointer=None,
    model_name: str = "gpt-4o",
):
    """
    Builds the Financial Advisor supervisor agent by fetching tools
    dynamically from decoupled MCP servers over Streamable HTTP.
    """
    if checkpointer is None:
        checkpointer = _get_postgres_checkpointer()

    openai_model = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    # 1. Fetch remote tools dynamically from all running MCP servers
    mcp_tools = client.get_tools()
    logger.info("Connected to MCP servers. Loaded %d remote tools.", len(mcp_tools))

    # 2. Partition tools to subagents
    tools_by_role = _partition_mcp_tools(mcp_tools)

    # 3. Assemble subagents and supervisor
    subagents = _build_subagents(openai_model, tools_by_role)

    return create_deep_agent(
        model=openai_model,
        tools=tools_by_role["supervisor"],
        system_prompt=load_prompt("financial_advisor"),
        subagents=subagents,
        checkpointer=checkpointer,
    )


class FinancialAdvisor:
    """
    Decoupled Financial Advisor Interface.
    Communicates with external ML processes over MCP (Streamable HTTP).
    """

    def __init__(self, checkpointer=None, model_name: str = "gpt-4o") -> None:
        self._checkpointer = checkpointer or _get_postgres_checkpointer()
        self._model_name = model_name
        self._mcp_servers = MCP_SERVER_CONFIG

    async def ask_async(
        self, question: str, thread_id: str = "default", recursion_limit: int = 25
    ) -> str:
        """Asynchronously connects to MCP servers, invokes agent, and returns response."""
        async with MultiServerMCPClient(self._mcp_servers) as mcp_client:
            agent = await build_financial_advisor_agent_async(
                mcp_client,
                checkpointer=self._checkpointer,
                model_name=self._model_name,
            )

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            }

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]}, config
            )
            messages = result.get("messages", [])
            
            logger.info("Agent returned %d messages", len(messages))
            return extract_response_text(messages)

    def ask(
        self, question: str, thread_id: str = "default", recursion_limit: int = 25
    ) -> str:
        """Synchronous wrapper for ask_async."""
        return asyncio.run(
            self.ask_async(
                question=question,
                thread_id=thread_id,
                recursion_limit=recursion_limit,
            )
        )