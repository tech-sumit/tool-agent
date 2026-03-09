"""MCP (Model Context Protocol) server.

Exposes all registered tools as MCP tools via FastMCP,
and provides the n8n node catalog as an MCP resource.
Mounts onto the main FastAPI app at /mcp.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from agent.router import ToolRouter
    from agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def create_mcp_server(
    registry: ToolRegistry,
    router: ToolRouter,
    agent_name: str = "Tool Agent",
) -> FastMCP:
    """Create a FastMCP server with tools from the registry."""

    mcp = FastMCP(
        agent_name,
        stateless_http=True,
    )

    @mcp.tool()
    async def list_tools(category: str = "") -> str:
        """List all available tools, optionally filtered by category.

        Args:
            category: Filter by category (integration, data, ai, utility, workflow). Empty for all.
        """
        tools = registry.list_tools()
        if category:
            tools = [t for t in tools if t["category"] == category]
        return json.dumps(tools, indent=2)

    @mcp.tool()
    async def search_tools(query: str) -> str:
        """Search for tools by keyword in names, descriptions, and tags.

        Args:
            query: Search query string.
        """
        results = registry.search(query)
        return json.dumps(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "category": s.category.value,
                    "tags": s.tags,
                }
                for s in results
            ],
            indent=2,
        )

    @mcp.tool()
    async def get_tool_schema(tool_name: str) -> str:
        """Get the full JSON Schema for a specific tool.

        Args:
            tool_name: Name of the tool to get schema for.
        """
        schema = registry.get_schema(tool_name)
        if not schema:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})
        return json.dumps(schema.to_function_schema(), indent=2)

    @mcp.tool()
    async def route_request(message: str) -> str:
        """Route a natural-language request to the appropriate tool(s) and execute.

        Uses FunctionGemma to select and call the best tool for the request.

        Args:
            message: Natural-language description of what you want to do.
        """
        result = await router.route(message=message, execute=True)
        return json.dumps(result.to_dict(), indent=2, default=str)

    @mcp.tool()
    async def execute_tool(tool_name: str, parameters: str = "{}") -> str:
        """Execute a specific tool directly with given parameters.

        Args:
            tool_name: Name of the tool to execute.
            parameters: JSON string of parameters to pass to the tool.
        """
        try:
            params = json.loads(parameters)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in parameters"})

        result = await registry.execute(tool_name, **params)
        return json.dumps(result.model_dump(), indent=2, default=str)

    @mcp.resource("tools://catalog")
    async def tool_catalog() -> str:
        """Complete catalog of all registered tools with schemas."""
        catalog = {}
        for name in registry.tool_names:
            schema = registry.get_schema(name)
            if schema:
                catalog[name] = {
                    "description": schema.description,
                    "category": schema.category.value,
                    "parameters": [p.model_dump() for p in schema.parameters],
                    "tags": schema.tags,
                }
        return json.dumps(catalog, indent=2)

    @mcp.resource("tools://capabilities")
    async def agent_capabilities() -> str:
        """Summary of agent capabilities and protocol endpoints."""
        return json.dumps(
            {
                "name": agent_name,
                "protocols": ["a2a", "mcp", "websocket"],
                "total_tools": registry.tool_count,
                "tool_names": registry.tool_names,
                "endpoints": {
                    "a2a": "/.well-known/agent-card.json",
                    "mcp": "/mcp",
                    "websocket": "/ws",
                    "health": "/health",
                    "tools": "/tools",
                },
            },
            indent=2,
        )

    # Dynamically register each tool from the registry as an MCP tool
    for tool_name in registry.tool_names:
        _register_dynamic_tool(mcp, registry, tool_name)

    return mcp


def _register_dynamic_tool(
    mcp: FastMCP,
    registry: ToolRegistry,
    tool_name: str,
) -> None:
    """Register a single registry tool as a native MCP tool."""
    schema = registry.get_schema(tool_name)
    if not schema:
        return

    # Prefix dynamic tools to avoid collision with built-in MCP tools above
    mcp_name = f"run_{tool_name}"

    async def _execute(**kwargs: Any) -> str:
        result = await registry.execute(tool_name, **kwargs)
        return json.dumps(result.model_dump(), indent=2, default=str)

    _execute.__name__ = mcp_name
    _execute.__doc__ = schema.description

    mcp_tool_decorator = mcp.tool(name=mcp_name, description=schema.description)
    mcp_tool_decorator(_execute)
