"""WebSocket protocol handler.

Provides a real-time WebSocket endpoint for tool execution with
JSON-RPC-style messages aligned with the A2A message format.

Message format (client -> server):
{
    "jsonrpc": "2.0",
    "id": "req-1",
    "method": "route" | "execute" | "list_tools" | "search_tools",
    "params": { ... }
}

Response format (server -> client):
{
    "jsonrpc": "2.0",
    "id": "req-1",
    "result": { ... }
}
or
{
    "jsonrpc": "2.0",
    "id": "req-1",
    "error": { "code": -32600, "message": "..." }
}
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from agent.composer import ToolComposer
    from agent.router import ToolRouter
    from agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages active WebSocket sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = websocket
        logger.info("WebSocket session connected: %s", session_id)
        return session_id

    def disconnect(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        logger.info("WebSocket session disconnected: %s", session_id)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


class WebSocketHandler:
    """Handles WebSocket JSON-RPC messages."""

    def __init__(
        self,
        registry: ToolRegistry,
        router: ToolRouter,
        composer: ToolComposer,
    ):
        self.registry = registry
        self.router = router
        self.composer = composer
        self.sessions = SessionManager()

    async def handle_connection(self, websocket: WebSocket) -> None:
        session_id = await self.sessions.connect(websocket)

        try:
            await self._send_welcome(websocket, session_id)

            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, None, -32700, "Parse error")
                    continue

                await self._dispatch(websocket, message)

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.exception("WebSocket error in session %s", session_id)
            try:
                await self._send_error(websocket, None, -32603, str(exc))
            except Exception:
                pass
        finally:
            self.sessions.disconnect(session_id)

    async def _send_welcome(self, ws: WebSocket, session_id: str) -> None:
        await ws.send_json({
            "jsonrpc": "2.0",
            "method": "connected",
            "params": {
                "session_id": session_id,
                "agent": "Tool Agent",
                "protocols": ["jsonrpc"],
                "tools_available": self.registry.tool_count,
            },
        })

    async def _dispatch(self, ws: WebSocket, message: dict) -> None:
        req_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params", {})

        handlers = {
            "route": self._handle_route,
            "compose": self._handle_compose,
            "execute": self._handle_execute,
            "list_tools": self._handle_list_tools,
            "search_tools": self._handle_search_tools,
            "get_schema": self._handle_get_schema,
        }

        handler = handlers.get(method)
        if not handler:
            await self._send_error(ws, req_id, -32601, f"Method not found: {method}")
            return

        try:
            result = await handler(params)
            await self._send_result(ws, req_id, result)
        except Exception as exc:
            logger.exception("Handler error for method %s", method)
            await self._send_error(ws, req_id, -32603, str(exc))

    async def _handle_route(self, params: dict) -> dict:
        message = params.get("message", "")
        if not message:
            return {"error": "Missing 'message' parameter"}

        result = await self.router.route(
            message=message,
            execute=params.get("execute", True),
        )
        return result.to_dict()

    async def _handle_compose(self, params: dict) -> dict:
        message = params.get("message", "")
        if not message:
            return {"error": "Missing 'message' parameter"}

        result = await self.composer.compose(message=message)
        return result.to_dict()

    async def _handle_execute(self, params: dict) -> dict:
        tool_name = params.get("tool")
        if not tool_name:
            return {"error": "Missing 'tool' parameter"}

        tool_params = params.get("parameters", {})
        result = await self.registry.execute(tool_name, **tool_params)
        return result.model_dump()

    async def _handle_list_tools(self, params: dict) -> dict:
        category = params.get("category", "")
        tools = self.registry.list_tools()
        if category:
            tools = [t for t in tools if t["category"] == category]
        return {"tools": tools, "total": len(tools)}

    async def _handle_search_tools(self, params: dict) -> dict:
        query = params.get("query", "")
        if not query:
            return {"error": "Missing 'query' parameter"}

        results = self.registry.search(query)
        return {
            "results": [
                {"name": s.name, "description": s.description, "tags": s.tags}
                for s in results
            ],
            "total": len(results),
        }

    async def _handle_get_schema(self, params: dict) -> dict:
        tool_name = params.get("tool", "")
        if not tool_name:
            return {"error": "Missing 'tool' parameter"}

        schema = self.registry.get_schema(tool_name)
        if not schema:
            return {"error": f"Tool '{tool_name}' not found"}

        return schema.to_function_schema()

    async def _send_result(self, ws: WebSocket, req_id: Any, result: Any) -> None:
        await ws.send_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        })

    async def _send_error(
        self, ws: WebSocket, req_id: Any, code: int, message: str
    ) -> None:
        await ws.send_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        })
