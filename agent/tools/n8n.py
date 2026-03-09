"""n8n integration tools — workflow CRUD and node operations.

These tools connect to the existing n8n instance via its REST API,
allowing the agent to list, create, trigger, and manage workflows.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from agent.config import config
from agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    ToolSchema,
)

logger = logging.getLogger(__name__)


class N8nClient:
    """Async client for the n8n REST API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or config.n8n_api_url).rstrip("/")
        self.api_key = api_key or config.n8n_api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-N8N-API-KEY": self.api_key} if self.api_key else {},
            timeout=30.0,
        )

    async def request(self, method: str, path: str, **kwargs) -> dict:
        resp = await self._client.request(method, f"/api/v1{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()


_n8n_client: N8nClient | None = None


def get_n8n_client() -> N8nClient:
    global _n8n_client
    if _n8n_client is None:
        _n8n_client = N8nClient()
    return _n8n_client


class ListWorkflowsTool(BaseTool):
    schema = ToolSchema(
        name="n8n_list_workflows",
        description="List all workflows in the n8n instance with their status",
        category=ToolCategory.WORKFLOW,
        parameters=[
            ToolParameter(name="active", type="boolean", description="Filter by active status", required=False),
            ToolParameter(name="limit", type="integer", description="Max workflows to return", required=False, default=50),
        ],
        tags=["n8n", "workflow", "list"],
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = get_n8n_client()
            params: dict[str, Any] = {}
            if "active" in kwargs:
                params["active"] = kwargs["active"]
            if "limit" in kwargs:
                params["limit"] = kwargs["limit"]

            data = await client.request("GET", "/workflows", params=params)
            workflows = data.get("data", data)
            summary = [
                {
                    "id": w.get("id"),
                    "name": w.get("name"),
                    "active": w.get("active"),
                    "nodes": len(w.get("nodes", [])),
                }
                for w in (workflows if isinstance(workflows, list) else [workflows])
            ]
            return ToolResult(success=True, data=summary)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GetWorkflowTool(BaseTool):
    schema = ToolSchema(
        name="n8n_get_workflow",
        description="Get detailed information about a specific n8n workflow by ID",
        category=ToolCategory.WORKFLOW,
        parameters=[
            ToolParameter(name="workflow_id", type="string", description="Workflow ID", required=True),
        ],
        tags=["n8n", "workflow", "details"],
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        workflow_id = kwargs.get("workflow_id")
        if not workflow_id:
            return ToolResult(success=False, error="workflow_id is required")
        try:
            client = get_n8n_client()
            data = await client.request("GET", f"/workflows/{workflow_id}")
            return ToolResult(success=True, data=data)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class TriggerWorkflowTool(BaseTool):
    schema = ToolSchema(
        name="n8n_trigger_workflow",
        description="Trigger/execute an n8n workflow by ID, optionally with input data",
        category=ToolCategory.WORKFLOW,
        parameters=[
            ToolParameter(name="workflow_id", type="string", description="Workflow ID to trigger", required=True),
            ToolParameter(name="data", type="object", description="Input data to pass to the workflow", required=False),
        ],
        tags=["n8n", "workflow", "trigger", "execute"],
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        workflow_id = kwargs.get("workflow_id")
        if not workflow_id:
            return ToolResult(success=False, error="workflow_id is required")
        try:
            client = get_n8n_client()
            body: dict[str, Any] = {}
            if "data" in kwargs:
                body["data"] = kwargs["data"]

            data = await client.request(
                "POST",
                f"/workflows/{workflow_id}/run",
                json=body if body else None,
            )
            return ToolResult(success=True, data=data)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class ToggleWorkflowTool(BaseTool):
    schema = ToolSchema(
        name="n8n_toggle_workflow",
        description="Enable or disable an n8n workflow",
        category=ToolCategory.WORKFLOW,
        parameters=[
            ToolParameter(name="workflow_id", type="string", description="Workflow ID", required=True),
            ToolParameter(name="active", type="boolean", description="True to enable, False to disable", required=True),
        ],
        tags=["n8n", "workflow", "enable", "disable"],
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        workflow_id = kwargs.get("workflow_id")
        active = kwargs.get("active")
        if not workflow_id:
            return ToolResult(success=False, error="workflow_id is required")
        if active is None:
            return ToolResult(success=False, error="active is required")
        try:
            client = get_n8n_client()
            data = await client.request(
                "PATCH",
                f"/workflows/{workflow_id}",
                json={"active": active},
            )
            return ToolResult(success=True, data={"id": workflow_id, "active": active})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class GetExecutionsTool(BaseTool):
    schema = ToolSchema(
        name="n8n_get_executions",
        description="Get recent workflow executions, optionally filtered by workflow ID",
        category=ToolCategory.WORKFLOW,
        parameters=[
            ToolParameter(name="workflow_id", type="string", description="Filter by workflow ID", required=False),
            ToolParameter(name="limit", type="integer", description="Max results", required=False, default=10),
            ToolParameter(name="status", type="string", description="Filter by status: success, error, waiting", required=False),
        ],
        tags=["n8n", "workflow", "executions", "logs"],
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            client = get_n8n_client()
            params: dict[str, Any] = {"limit": kwargs.get("limit", 10)}
            if "workflow_id" in kwargs:
                params["workflowId"] = kwargs["workflow_id"]
            if "status" in kwargs:
                params["status"] = kwargs["status"]

            data = await client.request("GET", "/executions", params=params)
            return ToolResult(success=True, data=data.get("data", data))
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class SearchWorkflowsTool(BaseTool):
    schema = ToolSchema(
        name="n8n_search_workflows",
        description="Search workflows by name",
        category=ToolCategory.WORKFLOW,
        parameters=[
            ToolParameter(name="query", type="string", description="Search query for workflow names", required=True),
        ],
        tags=["n8n", "workflow", "search"],
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="query is required")
        try:
            client = get_n8n_client()
            data = await client.request("GET", "/workflows")
            workflows = data.get("data", data)
            if isinstance(workflows, list):
                matches = [
                    {"id": w.get("id"), "name": w.get("name"), "active": w.get("active")}
                    for w in workflows
                    if query.lower() in (w.get("name") or "").lower()
                ]
            else:
                matches = []
            return ToolResult(success=True, data=matches)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


def register_n8n_tools(registry) -> None:
    """Register all n8n tools with the given registry."""
    tools = [
        ListWorkflowsTool(),
        GetWorkflowTool(),
        TriggerWorkflowTool(),
        ToggleWorkflowTool(),
        GetExecutionsTool(),
        SearchWorkflowsTool(),
    ]
    for tool in tools:
        registry.register(tool)
