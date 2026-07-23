"""
Shankh: AI-Assisted Financial Research Assistant for Indian Equity Markets.

Main entry point for CLI query execution and MCP server launching.
"""

import argparse
import logging
from src.shankh.agents.financial_advisor import FinancialAdvisor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("shankh.main")


def run_cli_query(question: str, thread_id: str = "main-cli"):
    print("=" * 80)
    print(" SHANKH FINANCIAL ADVISOR RESEARCH AGENT")
    print("=" * 80)
    print(f" Query     : {question}")
    print(f" Thread ID : {thread_id}")
    print("-" * 80)
    print(" Executing research agent workflow...\n")

    advisor = FinancialAdvisor()
    response = advisor.ask(question, thread_id=thread_id)

    print("=" * 80)
    print(" RESEARCH RESPONSE")
    print("=" * 80)
    print(response)
    print("=" * 80)


def start_mcp_server(transport: str = "streamable-http", host: str = "0.0.0.0", port: int = 8000):
    from src.shankh.mcp.server import app
    logger.info(f"Starting Shankh Financial Advisor MCP Server on {host}:{port} via {transport}...")
    app.run(transport=transport)


def main():
    parser = argparse.ArgumentParser(description="Shankh Financial Research Assistant")
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single financial research query using the Financial Advisor agent."
    )
    parser.add_argument(
        "--thread-id", "-t",
        type=str,
        default="default",
        help="Thread ID for agent conversation memory."
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Start the FastMCP HTTP SSE Server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for MCP Server (default: 8000)."
    )

    args = parser.parse_args()

    if args.mcp:
        start_mcp_server(port=args.port)
    elif args.query:
        run_cli_query(args.query, thread_id=args.thread_id)
    else:
        # Default interactive run if no arguments passed
        default_query = "Provide a comprehensive snapshot of current Indian market conditions and macro backdrop."
        run_cli_query(default_query, thread_id=args.thread_id)


if __name__ == "__main__":
    main()
