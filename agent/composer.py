"""Multi-step tool chain composer.

Builds on the router to support workflows that require sequential
tool execution, where output from one step feeds into the next.

Uses xLAM-2's multi-turn capability: after each tool execution,
the result is fed back to the model to decide the next action.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent.model import FunctionCall, InferenceBackend, OllamaBackend
from agent.router import ToolRouter
from agent.tool_registry import ToolRegistry
from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class CompositionStep:
    step_number: int
    function_call: FunctionCall
    result: ToolResult | None = None


@dataclass
class CompositionResult:
    """Result of a multi-step tool composition."""

    user_message: str
    steps: list[CompositionStep] = field(default_factory=list)
    final_result: Any = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and all(
            s.result is not None and s.result.success for s in self.steps
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "steps": [
                {
                    "step": s.step_number,
                    "tool": s.function_call.to_dict(),
                    "result": s.result.model_dump() if s.result else None,
                }
                for s in self.steps
            ],
            "final_result": self.final_result,
            "success": self.success,
            "error": self.error,
        }


class ToolComposer:
    """Composes multi-step tool chains using iterative model calls."""

    def __init__(
        self,
        registry: ToolRegistry,
        backend: InferenceBackend,
        max_steps: int = 10,
    ):
        self.registry = registry
        self.backend = backend
        self.router = ToolRouter(registry, backend)
        self.max_steps = max_steps

    async def compose(
        self,
        message: str,
        tool_names: list[str] | None = None,
    ) -> CompositionResult:
        """Execute a multi-step composition with multi-turn model interaction.

        For each step:
          1. Model produces tool call(s)
          2. Tools are executed
          3. Results are fed back to the model as context
          4. Model decides next action or terminates
        """
        result = CompositionResult(user_message=message)
        schemas = self.registry.get_function_schemas(tool_names)

        if not schemas:
            result.error = "No tools available for composition"
            return result

        conversation: list[dict[str, str]] = [
            {"role": "user", "content": message},
        ]

        for step_num in range(1, self.max_steps + 1):
            try:
                if isinstance(self.backend, OllamaBackend):
                    raw_output = await self.backend.generate_multi_turn(
                        messages=conversation,
                        tools=schemas,
                    )
                else:
                    context = message
                    for s in result.steps:
                        if s.result:
                            context += (
                                f"\nTool result from {s.function_call.name}: "
                                f"{json.dumps(s.result.data)}"
                            )
                    raw_output = await self.backend.generate(
                        user_message=context,
                        tools=schemas,
                    )
            except Exception as exc:
                logger.exception("Inference failed at step %d", step_num)
                result.error = f"Inference error at step {step_num}: {exc}"
                return result

            function_calls = FunctionCall.parse(raw_output)

            if not function_calls:
                if result.steps:
                    result.final_result = raw_output
                else:
                    result.error = "Model did not produce any function calls"
                return result

            conversation.append({"role": "assistant", "content": raw_output})

            for fc in function_calls:
                step = CompositionStep(
                    step_number=step_num,
                    function_call=fc,
                )

                tool_result = await self.registry.execute(fc.name, **fc.arguments)
                step.result = tool_result
                result.steps.append(step)

                if not tool_result.success:
                    result.error = (
                        f"Step {step_num} ({fc.name}) failed: {tool_result.error}"
                    )
                    return result

            last_result_data = json.dumps(
                [s.result.data for s in result.steps if s.step_number == step_num
                 and s.result],
            )
            conversation.append({
                "role": "user",
                "content": f"Tool result: {last_result_data}",
            })

        if result.steps:
            last_step = result.steps[-1]
            if last_step.result:
                result.final_result = last_step.result.data

        return result
