"""Tool router — maps natural-language requests to tool calls via FunctionGemma.

The router:
  1. Takes a user message and the available tool schemas
  2. Sends them to FunctionGemma for tool selection
  3. Parses the structured output into FunctionCall objects
  4. Executes the selected tool(s) through the ToolRegistry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent.model import FunctionCall, InferenceBackend
from agent.tool_registry import ToolRegistry
from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of routing a user request through FunctionGemma."""

    user_message: str
    raw_output: str
    function_calls: list[FunctionCall]
    results: list[ToolResult] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and all(r.success for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "raw_output": self.raw_output,
            "function_calls": [fc.to_dict() for fc in self.function_calls],
            "results": [r.model_dump() for r in self.results],
            "success": self.success,
            "error": self.error,
        }


class ToolRouter:
    """Routes natural-language requests to tools via FunctionGemma."""

    def __init__(
        self,
        registry: ToolRegistry,
        backend: InferenceBackend,
    ):
        self.registry = registry
        self.backend = backend

    async def route(
        self,
        message: str,
        tool_names: list[str] | None = None,
        execute: bool = True,
        max_tokens: int = 512,
    ) -> RoutingResult:
        """Route a message to the appropriate tool(s).

        Args:
            message: Natural-language user request.
            tool_names: Restrict to these tools (None = all registered).
            execute: Whether to actually execute the tools or just select them.
            max_tokens: Max generation tokens.
        """
        schemas = self.registry.get_function_schemas(tool_names)

        if not schemas:
            return RoutingResult(
                user_message=message,
                raw_output="",
                function_calls=[],
                error="No tools available for routing",
            )

        try:
            raw_output = await self.backend.generate(
                user_message=message,
                tools=schemas,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.exception("Model inference failed")
            return RoutingResult(
                user_message=message,
                raw_output="",
                function_calls=[],
                error=f"Inference error: {exc}",
            )

        function_calls = FunctionCall.parse(raw_output)

        if not function_calls:
            return RoutingResult(
                user_message=message,
                raw_output=raw_output,
                function_calls=[],
                error="Model did not produce any function calls",
            )

        results: list[ToolResult] = []
        if execute:
            for fc in function_calls:
                result = await self.registry.execute(fc.name, **fc.arguments)
                results.append(result)

        return RoutingResult(
            user_message=message,
            raw_output=raw_output,
            function_calls=function_calls,
            results=results,
        )
