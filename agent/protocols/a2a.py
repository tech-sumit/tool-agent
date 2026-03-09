"""A2A (Agent-to-Agent) protocol server.

Implements the Google A2A protocol using the official a2a-sdk.
Serves an AgentCard at /.well-known/agent-card.json and handles
task execution via JSON-RPC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.message import new_agent_text_message

if TYPE_CHECKING:
    from fastapi import FastAPI

    from agent.composer import ToolComposer
    from agent.router import ToolRouter
    from agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolAgentExecutor(AgentExecutor):
    """A2A executor that delegates to the ToolRouter/Composer."""

    def __init__(self, router: ToolRouter, composer: ToolComposer):
        self.router = router
        self.composer = composer

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_text = ""
        if context.message and context.message.parts:
            part = context.message.parts[0]
            if hasattr(part, "root") and hasattr(part.root, "text"):
                user_text = part.root.text
            elif hasattr(part, "text"):
                user_text = part.text

        if not user_text:
            await event_queue.put(
                Task(
                    id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            "No text input provided.",
                            context.context_id,
                            context.task_id,
                        ),
                    ),
                )
            )
            return

        await event_queue.put(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        try:
            result = await self.router.route(message=user_text, execute=True)

            if result.success:
                import json

                response_data = result.to_dict()
                response_text = json.dumps(response_data, indent=2, default=str)

                await event_queue.put(
                    Task(
                        id=context.task_id,
                        context_id=context.context_id,
                        status=TaskStatus(
                            state=TaskState.completed,
                            message=new_agent_text_message(
                                response_text,
                                context.context_id,
                                context.task_id,
                            ),
                        ),
                    )
                )
            else:
                await event_queue.put(
                    Task(
                        id=context.task_id,
                        context_id=context.context_id,
                        status=TaskStatus(
                            state=TaskState.failed,
                            message=new_agent_text_message(
                                f"Routing failed: {result.error}",
                                context.context_id,
                                context.task_id,
                            ),
                        ),
                    )
                )

        except Exception as exc:
            logger.exception("A2A execution failed")
            await event_queue.put(
                Task(
                    id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            f"Internal error: {exc}",
                            context.context_id,
                            context.task_id,
                        ),
                    ),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        await event_queue.put(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.canceled),
            )
        )


def build_agent_card(
    registry: ToolRegistry,
    agent_name: str,
    agent_url: str,
    description: str,
    version: str,
) -> AgentCard:
    """Build an A2A AgentCard from the tool registry."""
    skills = []
    for tool_info in registry.list_tools():
        skills.append(
            AgentSkill(
                id=tool_info["name"],
                name=tool_info["name"].replace("_", " ").title(),
                description=tool_info["description"],
                tags=tool_info.get("tags", []),
            )
        )

    if not skills:
        skills.append(
            AgentSkill(
                id="general",
                name="General Tool Routing",
                description="Route natural-language requests to appropriate integration tools",
                tags=["routing", "tools"],
            )
        )

    return AgentCard(
        name=agent_name,
        description=description,
        url=agent_url,
        version=version,
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
        ),
        skills=skills,
        defaultInputModes=["text/plain", "application/json"],
        defaultOutputModes=["application/json", "text/plain"],
    )


def mount_a2a(
    app: FastAPI,
    registry: ToolRegistry,
    router: ToolRouter,
    composer: ToolComposer,
    agent_name: str,
    agent_url: str,
    description: str,
    version: str,
) -> FastAPI:
    """Mount the A2A protocol onto an existing FastAPI app."""
    agent_card = build_agent_card(
        registry=registry,
        agent_name=agent_name,
        agent_url=agent_url,
        description=description,
        version=version,
    )

    executor = ToolAgentExecutor(router=router, composer=composer)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AFastAPIApplication(agent_card, handler)
    a2a_fastapi = a2a_app.build()

    app.mount("/a2a", a2a_fastapi)

    @app.get("/.well-known/agent-card.json")
    async def get_agent_card():
        return agent_card.model_dump(by_alias=True, exclude_none=True)

    return app
