"""MCP client bridge — connects to external MCP servers and registers their tools.

Allows the Tool Agent to consume tools from any MCP-compatible server
(Firecrawl, filesystem, database, etc.) and expose them through all
its own protocols (REST, WebSocket, A2A, MCP).

Usage:
    bridge = McpToolBridge(
        command="npx",
        args=["-y", "firecrawl-mcp"],
        env={"FIRECRAWL_API_KEY": "..."},
    )

    async with bridge:
        bridge.register_tools(registry)
        # ... tools are now available for routing
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    ToolSchema,
)

logger = logging.getLogger(__name__)


def _schema_to_params(input_schema: dict) -> list[ToolParameter]:
    """Convert a JSON Schema properties dict to a ToolParameter list."""
    params = []
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    for name, prop in properties.items():
        params.append(ToolParameter(
            name=name,
            type=prop.get("type", "string"),
            description=prop.get("description", ""),
            required=name in required,
            default=prop.get("default"),
            enum=prop.get("enum"),
        ))

    return params


class McpTool(BaseTool):
    """A tool backed by an external MCP server session."""

    def __init__(
        self,
        tool_name: str,
        description: str,
        input_schema: dict,
        session: ClientSession,
        category: ToolCategory = ToolCategory.UTILITY,
        tags: list[str] | None = None,
    ):
        self.schema = ToolSchema(
            name=tool_name,
            description=description,
            category=category,
            parameters=_schema_to_params(input_schema),
            tags=tags or [],
        )
        self._session = session

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._session.call_tool(self.schema.name, kwargs)

            text_parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)

            combined = "\n".join(text_parts)

            try:
                data = json.loads(combined)
            except (json.JSONDecodeError, TypeError):
                data = combined

            return ToolResult(success=not result.isError, data=data)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class McpToolBridge:
    """Connects to an external MCP server via stdio and bridges its tools
    into the agent's ToolRegistry.

    Manages the subprocess lifecycle — use as an async context manager:

        async with McpToolBridge("npx", ["-y", "some-mcp"]) as bridge:
            bridge.register_tools(registry)
            # subprocess stays alive; tools are callable
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        category: ToolCategory = ToolCategory.UTILITY,
        tags: list[str] | None = None,
    ):
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )
        self.category = category
        self.tags = tags or []
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tools: list[McpTool] = []

    async def connect(self) -> list[McpTool]:
        """Start the MCP server subprocess and discover its tools."""
        self._stack = AsyncExitStack()

        transport = await self._stack.enter_async_context(
            stdio_client(self.server_params)
        )
        read_stream, write_stream = transport

        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        tools_result = await self._session.list_tools()

        self._tools = []
        for tool in tools_result.tools:
            schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
            mcp_tool = McpTool(
                tool_name=tool.name,
                description=tool.description or "",
                input_schema=schema,
                session=self._session,
                category=self.category,
                tags=self.tags,
            )
            self._tools.append(mcp_tool)
            logger.info("Bridged MCP tool: %s", tool.name)

        logger.info("Connected to MCP server — %d tools discovered", len(self._tools))
        return self._tools

    @property
    def tools(self) -> list[McpTool]:
        return list(self._tools)

    def register_tools(self, registry) -> int:
        """Register all discovered tools in the agent's ToolRegistry."""
        for tool in self._tools:
            registry.register(tool)
        return len(self._tools)

    async def disconnect(self):
        """Shut down the MCP server subprocess."""
        if self._stack:
            await self._stack.aclose()
            self._stack = None
            self._session = None
            self._tools = []

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.disconnect()
