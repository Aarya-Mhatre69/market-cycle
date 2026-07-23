"""
Production-ready MCP Server exposing the Shankh Financial Advisor agent.

This module initializes a FastMCP server and provides a tool to query the Financial Advisor.
It is designed to be exposed via HTTP SSE (Server-Sent Events) to provide a streamable
connection for MCP clients.
"""

import logging
import os

from fastmcp import FastMCP
from src.shankh.agents.financial_advisor import FinancialAdvisor

from src.shankh.utils import extract_text_content

# Configure logging for production readiness
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the FastMCP server
app = FastMCP("ShankhFinancialAdvisor")

# Initialize the underlying agent
try:
    advisor = FinancialAdvisor()
    logger.info("Successfully initialized the Financial Advisor agent.")
except Exception:
    logger.exception("Failed to initialize the Financial Advisor agent.")
    raise


@app.tool()
def ask_financial_advisor(question: str, thread_id: str = "default") -> str:
    """
    Consult the Shankh Financial Advisor for insights on market regimes, macroeconomic indicators,
    stock performance, sector rotations, and general investment queries.

    This tool delegates the question to the internal agent graph, which orchestrates various
    market data tools and sub-agents to synthesize a comprehensive financial analysis.

    Args:
        question (str): The user's financial query or topic of interest.
        thread_id (str, optional): A unique identifier for the conversation thread to maintain
                                   memory and context across multiple turns. Defaults to "default".

    Returns:
        str: The detailed response from the financial advisor, formatted in Markdown.
    """
    logger.info(f"Processing query on thread '{thread_id}': {question}")
    try:
        # Invoke the agent graph with the provided question and thread context
        response = advisor.ask(question, thread_id=thread_id)
        text_response = extract_text_content(response)
        logger.info(f"Successfully generated response for thread '{thread_id}'.")
        return text_response
    except Exception as e:
        logger.exception(f"Error during agent invocation on thread '{thread_id}': {e}")
        return f"Error: Unable to consult the financial advisor at this time. Details: {str(e)}"


if __name__ == "__main__":
    # Retrieve configuration from environment with sensible defaults
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    
    logger.info(f"Starting Shankh Financial Advisor MCP Server on {host}:{port} via HTTP streamable-http...")
    
    app.run(transport="streamable-http")
