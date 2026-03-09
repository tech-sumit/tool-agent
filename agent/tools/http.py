"""Generic HTTP/API tools for making external requests."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    ToolSchema,
)

logger = logging.getLogger(__name__)


class HttpRequestTool(BaseTool):
    schema = ToolSchema(
        name="http_request",
        description="Make an HTTP request to any URL. Supports GET, POST, PUT, PATCH, DELETE with headers and body.",
        category=ToolCategory.UTILITY,
        parameters=[
            ToolParameter(name="url", type="string", description="Target URL", required=True),
            ToolParameter(
                name="method",
                type="string",
                description="HTTP method",
                required=False,
                default="GET",
                enum=["GET", "POST", "PUT", "PATCH", "DELETE"],
            ),
            ToolParameter(name="headers", type="object", description="HTTP headers as key-value pairs", required=False),
            ToolParameter(name="body", type="string", description="Request body (JSON string for POST/PUT)", required=False),
            ToolParameter(name="timeout", type="integer", description="Request timeout in seconds", required=False, default=30),
        ],
        tags=["http", "api", "request", "rest"],
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        url = kwargs.get("url")
        if not url:
            return ToolResult(success=False, error="url is required")

        method = kwargs.get("method", "GET").upper()
        headers = kwargs.get("headers", {})
        body = kwargs.get("body")
        timeout = kwargs.get("timeout", 30)

        if isinstance(headers, str):
            try:
                headers = json.loads(headers)
            except json.JSONDecodeError:
                headers = {}

        try:
            async with httpx.AsyncClient(timeout=float(timeout)) as client:
                request_kwargs: dict[str, Any] = {"headers": headers}

                if body and method in ("POST", "PUT", "PATCH"):
                    try:
                        request_kwargs["json"] = json.loads(body)
                    except (json.JSONDecodeError, TypeError):
                        request_kwargs["content"] = body

                resp = await client.request(method, url, **request_kwargs)

                try:
                    response_data = resp.json()
                except Exception:
                    response_data = resp.text

                return ToolResult(
                    success=True,
                    data={
                        "status_code": resp.status_code,
                        "headers": dict(resp.headers),
                        "body": response_data,
                    },
                )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class JsonTransformTool(BaseTool):
    schema = ToolSchema(
        name="json_transform",
        description="Transform JSON data by extracting fields, filtering, or restructuring",
        category=ToolCategory.DATA,
        parameters=[
            ToolParameter(name="data", type="string", description="Input JSON data as string", required=True),
            ToolParameter(name="expression", type="string", description="JSONPath-like expression or Python dict comprehension", required=True),
        ],
        tags=["json", "transform", "data", "parse"],
    )

    async def execute(self, **kwargs: Any) -> ToolResult:
        data_str = kwargs.get("data", "")
        expression = kwargs.get("expression", "")

        if not data_str or not expression:
            return ToolResult(success=False, error="Both 'data' and 'expression' are required")

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError as exc:
            return ToolResult(success=False, error=f"Invalid JSON: {exc}")

        try:
            parts = expression.split(".")
            result = data
            for part in parts:
                if isinstance(result, dict):
                    result = result.get(part, None)
                elif isinstance(result, list) and part.isdigit():
                    result = result[int(part)]
                else:
                    result = None
                    break

            return ToolResult(success=True, data=result)
        except Exception as exc:
            return ToolResult(success=False, error=f"Transform error: {exc}")


def register_http_tools(registry) -> None:
    """Register HTTP/utility tools with the registry."""
    tools = [
        HttpRequestTool(),
        JsonTransformTool(),
    ]
    for tool in tools:
        registry.register(tool)
