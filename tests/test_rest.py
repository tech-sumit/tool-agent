"""Integration tests for REST endpoints."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["tools_registered"] >= 2
    assert "a2a" in data["protocols"]
    assert "mcp" in data["protocols"]


@pytest.mark.asyncio
async def test_list_tools(client):
    resp = await client.get("/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 2
    names = {t["name"] for t in data["tools"]}
    assert "http_request" in names
    assert "json_transform" in names


@pytest.mark.asyncio
async def test_list_tools_search(client):
    resp = await client.get("/tools", params={"query": "http"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert any(t["name"] == "http_request" for t in data["tools"])


@pytest.mark.asyncio
async def test_get_tool_schema(client):
    resp = await client.get("/tools/http_request")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "function"
    assert data["function"]["name"] == "http_request"
    assert "parameters" in data["function"]


@pytest.mark.asyncio
async def test_get_tool_not_found(client):
    resp = await client.get("/tools/nonexistent_tool")
    assert resp.status_code == 200
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_route_message(client):
    resp = await client.post("/route", json={
        "message": "Make an http request to https://example.com",
        "execute": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "function_calls" in data
    assert len(data["function_calls"]) >= 1
    assert data["function_calls"][0]["name"] == "http_request"


@pytest.mark.asyncio
async def test_execute_tool(client):
    resp = await client.post("/execute", json={
        "tool": "json_transform",
        "parameters": {
            "data": '{"name": "test", "value": 42}',
            "expression": "name",
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"] == "test"


@pytest.mark.asyncio
async def test_execute_missing_tool(client):
    resp = await client.post("/execute", json={
        "tool": "nonexistent",
        "parameters": {},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "not found" in data["error"]
