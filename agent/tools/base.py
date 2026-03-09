"""Base tool interface and types for the tool agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    INTEGRATION = "integration"
    DATA = "data"
    AI = "ai"
    UTILITY = "utility"
    WORKFLOW = "workflow"


class ToolParameter(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    examples: list[str] = Field(default_factory=list)
    enum: list[str] | None = None


class ToolSchema(BaseModel):
    """JSON-Schema-style descriptor for a tool, used across all protocols."""

    name: str
    description: str
    category: ToolCategory = ToolCategory.UTILITY
    parameters: list[ToolParameter] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    def to_function_schema(self) -> dict[str, Any]:
        """Convert to FunctionGemma / OpenAI function-calling schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type}
            if param.description:
                prop["description"] = param.description
            if param.examples:
                prop["examples"] = param.examples
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }
        if required:
            schema["function"]["parameters"]["required"] = required

        return schema

    def to_a2a_skill(self) -> dict[str, Any]:
        """Convert to A2A AgentCard skill descriptor."""
        return {
            "id": self.name,
            "name": self.name.replace("_", " ").title(),
            "description": self.description,
            "inputModes": ["text/plain", "application/json"],
            "outputModes": ["application/json"],
            "tags": self.tags,
            "examples": [p.examples[0] for p in self.parameters if p.examples],
        }

    def to_mcp_tool(self) -> dict[str, Any]:
        """Convert to MCP tool input_schema format."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema


class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None


class BaseTool(ABC):
    """Abstract base for all tools registered with the agent."""

    schema: ToolSchema

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def description(self) -> str:
        return self.schema.description
