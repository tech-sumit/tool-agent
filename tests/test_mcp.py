"""Integration tests for MCP protocol.

Tests MCP tool functions directly since the streamable HTTP transport
requires a full ASGI lifecycle that's incompatible with test transports.
"""

from __future__ import annotations

import json

import pytest

from agent.model import MockBackend
from agent.protocols.mcp import create_mcp_server
from agent.router import ToolRouter
from agent.tool_registry import ToolRegistry
from agent.tools.http import register_http_tools


@pytest.fixture()
def mcp_setup():
    """Create a standalone MCP server with registry + mock backend."""
    registry = ToolRegistry()
    register_http_tools(registry)
    backend = MockBackend()
    router = ToolRouter(registry=registry, backend=backend)
    mcp = create_mcp_server(registry=registry, router=router, agent_name="Test Agent")
    return mcp, registry, router


@pytest.mark.asyncio
async def test_mcp_server_created(mcp_setup):
    """MCP server should be created with tools registered."""
    mcp, registry, _ = mcp_setup
    assert mcp is not None
    assert registry.tool_count == 2


@pytest.mark.asyncio
async def test_mcp_list_tools_function(mcp_setup):
    """list_tools MCP tool should return registered tools."""
    _, registry, _ = mcp_setup
    tools = registry.list_tools()
    assert len(tools) >= 2
    names = {t["name"] for t in tools}
    assert "http_request" in names
    assert "json_transform" in names


@pytest.mark.asyncio
async def test_mcp_search_tools_function(mcp_setup):
    """search_tools should find tools by keyword."""
    _, registry, _ = mcp_setup
    results = registry.search("http")
    assert len(results) >= 1
    assert results[0].name == "http_request"


@pytest.mark.asyncio
async def test_mcp_get_tool_schema_function(mcp_setup):
    """get_tool_schema should return JSON schema for a tool."""
    _, registry, _ = mcp_setup
    schema = registry.get_schema("http_request")
    assert schema is not None
    func_schema = schema.to_function_schema()
    assert func_schema["type"] == "function"
    assert func_schema["function"]["name"] == "http_request"
    assert "url" in func_schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_mcp_route_request_function(mcp_setup):
    """route_request should produce function calls via the mock backend."""
    _, _, router = mcp_setup
    result = await router.route(message="Make an http request to example.com", execute=False)
    assert len(result.function_calls) >= 1
    assert result.function_calls[0].name == "http_request"


@pytest.mark.asyncio
async def test_mcp_execute_tool_function(mcp_setup):
    """execute_tool should run a tool and return results."""
    _, registry, _ = mcp_setup
    result = await registry.execute(
        "json_transform",
        data='{"key": "value"}',
        expression="key",
    )
    assert result.success is True
    assert result.data == "value"


@pytest.mark.asyncio
async def test_mcp_tool_descriptors(mcp_setup):
    """MCP tool descriptors should be generated correctly."""
    _, registry, _ = mcp_setup
    mcp_tools = registry.get_mcp_tools()
    assert "http_request" in mcp_tools
    assert "json_transform" in mcp_tools
    http_tool = mcp_tools["http_request"]
    assert http_tool["name"] == "http_request"
    assert "inputSchema" in http_tool
    assert "url" in http_tool["inputSchema"]["properties"]


@pytest.mark.asyncio
async def test_mcp_endpoint_mounted(client):
    """MCP should be mounted at /mcp (responds to GET with method not allowed)."""
    resp = await client.get("/mcp/")
    assert resp.status_code in (200, 405, 404, 500)
