"""
Integration Tests for Main Research Assistant MCP Server.

Tests end-to-end execution of `query_research_assistant` over the FastMCP
client transport layer.

Run:
    RUN_LIVE_AGENT_TESTS=1 pytest tests/integration/test_mcp_main_server.py -v -s
"""

import json
import os
import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

if os.getenv("OPENAI_API_KEY"):
    pytest.skip(
        "Set OPENAI_API_KEY to run live MCP main server tests.",
        allow_module_level=True,
    )

from shankh.mcp.server import mcp as server_mcp


@pytest.fixture
async def main_mcp_client():
    """Fixture spinning up FastMCP client wrapping local Main Orchestrator MCP server."""
    async with Client(transport=server_mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_query_research_assistant_mcp_endpoint(
    main_mcp_client: Client[FastMCPTransport],
):
    """
    Test main MCP server by querying research assistant tool using FastMCP Client.
    Validates end-to-end multi-agent resolution and non-advisory compliance.
    """
    result = await main_mcp_client.call_tool(
        name="query_research_assistant",
        arguments={
            "question": "What is the current US-India 10Y yield spread and market regime?",
            "thread_id": "test-mcp-orchestrator",
        },
    )

    assert result is not None, "Expected result from MCP server tool call"

    if hasattr(result, "data") and result.data is not None:
        raw_text = str(result.data)
    elif hasattr(result, "content"):
        if isinstance(result.content, list):
            raw_text = "".join(
                str(b.text) if hasattr(b, "text") else str(b) for b in result.content
            )
        else:
            raw_text = str(result.content)
    else:
        raw_text = str(result)

    payload = json.loads(raw_text)
    assert payload.get("status") == "success"

    response_text = payload.get("response", "")
    assert len(response_text) > 200, "Response text should be detailed (>200 chars)"

    text_lower = response_text.lower()
    assert any(
        term in text_lower for term in ["yield", "spread", "regime", "percent"]
    ), f"Expected response to contain macro/regime markers. Got: {response_text}"