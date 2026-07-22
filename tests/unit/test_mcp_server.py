import pytest
import os
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from shankh.mcp.web.server import app as server_app

@pytest.fixture
async def main_mcp_client():
    """
    Fixture that spins up the FastMCP client wrapping our local MCP server.
    """
    async with Client(transport=server_app) as client:
        yield client

@pytest.mark.asyncio
async def test_ask_financial_advisor_nvda_price(main_mcp_client: Client[FastMCPTransport]):
    """
    Test the MCP server's financial advisor tool by querying the server using the FastMCP Client.
    This tests the actual MCP server workflow (tool resolution, parameter passing).
    
    Note: Requires appropriate environment variables (e.g., GOOGLE_API_KEY, TAVILY_API_KEY).
    """
    # Call the tool through the MCP client layer
    result = await main_mcp_client.call_tool(
        name="ask_financial_advisor", 
        arguments={
            "question": "What is the current price of NVDA stock?",
            "thread_id": "test-mcp-nvda"
        }
    )
    
    assert result is not None, "Expected a result from the MCP tool call"
    
    # FastMCP Client usually exposes the return value via `result.data` or `result.content`
    if hasattr(result, "data"):
        response_text = str(result.data)
    elif hasattr(result, "content"):
        # For standard MCP SDK compatibility
        if isinstance(result.content, list):
            response_text = "".join(
                str(block.text) if hasattr(block, "text") else str(block) 
                for block in result.content
            )
        else:
            response_text = str(result.content)
    else:
        response_text = str(result)
        
    assert len(response_text) > 0, "Response text should not be empty"
    
    response_upper = response_text.upper()
    assert "NVDA" in response_upper or "NVIDIA" in response_upper, \
        f"Expected response to mention NVDA or NVIDIA. Got: {response_text}"
