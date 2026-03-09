"""Shared test fixtures for integration tests.

Sets TOOL_AGENT_BACKEND=mock so all tests run without Ollama.
"""

from __future__ import annotations

import os

os.environ["TOOL_AGENT_BACKEND"] = "mock"
os.environ["N8N_API_KEY"] = ""

import pytest
from httpx import ASGITransport, AsyncClient

from agent.server import app


@pytest.fixture(scope="session", autouse=True)
async def _lifespan():
    """Trigger FastAPI lifespan so tools/backend are initialized."""
    async with app.router.lifespan_context(app):
        yield


@pytest.fixture()
async def client():
    """Async HTTP client wired to the FastAPI app."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
