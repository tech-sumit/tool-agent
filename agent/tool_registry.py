"""Dynamic tool registry with auto-discovery generation.

Central registry where tools are registered and looked up.
Automatically produces A2A AgentCard skills, MCP tool descriptors,
and FunctionGemma function schemas from the same source of truth.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.tools.base import BaseTool, ToolCategory, ToolResult, ToolSchema

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Thread-safe tool registry with dynamic registration."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._schemas: dict[str, ToolSchema] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance. Overwrites if name already exists."""
        name = tool.name
        self._tools[name] = tool
        self._schemas[name] = tool.schema
        logger.info("Registered tool: %s", name)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        removed = name in self._tools
        self._tools.pop(name, None)
        self._schemas.pop(name, None)
        if removed:
            logger.info("Unregistered tool: %s", name)
        return removed

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_schema(self, name: str) -> ToolSchema | None:
        return self._schemas.get(name)

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Look up and execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        try:
            return await tool.execute(**kwargs)
        except Exception as exc:
            logger.exception("Tool '%s' execution failed", name)
            return ToolResult(success=False, error=str(exc))

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return summary dicts for all registered tools."""
        return [
            {
                "name": schema.name,
                "description": schema.description,
                "category": schema.category.value,
                "parameters": len(schema.parameters),
                "tags": schema.tags,
            }
            for schema in self._schemas.values()
        ]

    def list_by_category(self, category: ToolCategory) -> list[ToolSchema]:
        return [s for s in self._schemas.values() if s.category == category]

    def search(self, query: str) -> list[ToolSchema]:
        """Simple keyword search across tool names, descriptions, and tags."""
        query_lower = query.lower()
        results = []
        for schema in self._schemas.values():
            text = f"{schema.name} {schema.description} {' '.join(schema.tags)}".lower()
            if query_lower in text:
                results.append(schema)
        return results

    # ── Protocol-specific exports ────────────────────────────────────

    def get_function_schemas(self, names: list[str] | None = None) -> list[dict]:
        """Get FunctionGemma-compatible function schemas."""
        schemas = self._schemas.values() if names is None else [
            self._schemas[n] for n in names if n in self._schemas
        ]
        return [s.to_function_schema() for s in schemas]

    def get_a2a_skills(self) -> list[dict]:
        """Get A2A AgentCard skill descriptors for all tools."""
        return [s.to_a2a_skill() for s in self._schemas.values()]

    def get_mcp_tools(self) -> dict[str, dict]:
        """Get MCP tool descriptors keyed by name."""
        return {
            name: {
                "name": schema.name,
                "description": schema.description,
                "inputSchema": schema.to_mcp_tool(),
            }
            for name, schema in self._schemas.items()
        }

    def get_agent_card(self, agent_name: str, agent_url: str, description: str, version: str) -> dict:
        """Generate a complete A2A AgentCard."""
        return {
            "name": agent_name,
            "description": description,
            "url": agent_url,
            "version": version,
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            },
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["application/json"],
            "skills": self.get_a2a_skills(),
        }
