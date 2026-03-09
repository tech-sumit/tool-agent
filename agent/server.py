"""Main FastAPI server — wires all protocols and services together.

Integrates:
  - A2A protocol at /a2a + /.well-known/agent-card.json
  - MCP protocol at /mcp
  - WebSocket at /ws
  - REST endpoints at /health, /tools, /route, /execute
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.composer import ToolComposer
from agent.config import config
from agent.model import InferenceBackend, create_backend
from agent.router import ToolRouter
from agent.tool_registry import ToolRegistry
from agent.tools.http import register_http_tools
from agent.tools.n8n import register_n8n_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

registry = ToolRegistry()
backend: InferenceBackend | None = None
router: ToolRouter | None = None
composer: ToolComposer | None = None
ws_handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend, router, composer, ws_handler

    logger.info("Starting Tool Agent v%s", config.agent_version)

    register_n8n_tools(registry)
    register_http_tools(registry)
    logger.info("Registered %d tools", registry.tool_count)

    backend = create_backend()
    router = ToolRouter(registry=registry, backend=backend)
    composer = ToolComposer(registry=registry, backend=backend)

    from agent.protocols.websocket import WebSocketHandler

    ws_handler = WebSocketHandler(
        registry=registry, router=router, composer=composer
    )

    _mount_protocols()

    model_healthy = await backend.health_check()
    if model_healthy:
        logger.info("Model backend (%s) is healthy", config.model_backend)
    else:
        logger.warning(
            "Model backend (%s) is not reachable — tool routing will fail until it's available",
            config.model_backend,
        )

    logger.info(
        "Agent ready on %s:%d — A2A, MCP, WebSocket, REST",
        config.host,
        config.port,
    )

    yield

    if hasattr(backend, "close"):
        await backend.close()
    logger.info("Tool Agent shut down.")


app = FastAPI(
    title=config.agent_name,
    description=config.agent_description,
    version=config.agent_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ───────────────────────────────────────────────────


@app.get("/health")
async def health():
    model_ok = await backend.health_check() if backend else False
    return {
        "status": "healthy" if model_ok else "degraded",
        "agent": config.agent_name,
        "version": config.agent_version,
        "model_backend": config.model_backend,
        "model_healthy": model_ok,
        "tools_registered": registry.tool_count,
        "protocols": ["a2a", "mcp", "websocket", "rest"],
        "ws_sessions": ws_handler.sessions.active_count if ws_handler else 0,
    }


@app.get("/tools")
async def list_tools(category: str = "", query: str = ""):
    if query:
        results = registry.search(query)
        return {
            "tools": [
                {
                    "name": s.name,
                    "description": s.description,
                    "category": s.category.value,
                    "tags": s.tags,
                }
                for s in results
            ],
            "total": len(results),
        }

    tools = registry.list_tools()
    if category:
        tools = [t for t in tools if t["category"] == category]
    return {"tools": tools, "total": len(tools)}


@app.get("/tools/{tool_name}")
async def get_tool(tool_name: str):
    schema = registry.get_schema(tool_name)
    if not schema:
        return {"error": f"Tool '{tool_name}' not found"}
    return schema.to_function_schema()


class RouteRequest(BaseModel):
    message: str
    execute: bool = True
    tool_names: list[str] | None = None


@app.post("/route")
async def route_message(req: RouteRequest):
    if not router:
        return {"error": "Router not initialized"}
    result = await router.route(
        message=req.message,
        execute=req.execute,
        tool_names=req.tool_names,
    )
    return result.to_dict()


class ComposeRequest(BaseModel):
    message: str
    tool_names: list[str] | None = None


@app.post("/compose")
async def compose_message(req: ComposeRequest):
    if not composer:
        return {"error": "Composer not initialized"}
    result = await composer.compose(
        message=req.message,
        tool_names=req.tool_names,
    )
    return result.to_dict()


class ExecuteRequest(BaseModel):
    tool: str
    parameters: dict[str, Any] = {}


@app.post("/execute")
async def execute_tool(req: ExecuteRequest):
    result = await registry.execute(req.tool, **req.parameters)
    return result.model_dump()


# ── WebSocket ────────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if ws_handler:
        await ws_handler.handle_connection(websocket)


# ── Protocol Mounting (deferred to avoid import cycles) ──────────────


def _mount_protocols():
    """Mount A2A and MCP after services are initialized."""
    from agent.protocols.a2a import mount_a2a
    from agent.protocols.mcp import create_mcp_server

    agent_url = f"http://{config.host}:{config.port}"

    mount_a2a(
        app=app,
        registry=registry,
        router=router,
        composer=composer,
        agent_name=config.agent_name,
        agent_url=agent_url,
        description=config.agent_description,
        version=config.agent_version,
    )

    mcp_server = create_mcp_server(
        registry=registry,
        router=router,
        agent_name=config.agent_name,
    )
    app.mount("/mcp", mcp_server.streamable_http_app())


def main():
    parser = argparse.ArgumentParser(description="Tool Agent Server")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    config.host = args.host
    config.port = args.port

    uvicorn.run(
        "agent.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
