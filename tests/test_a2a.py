"""Integration tests for A2A protocol."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_card(client):
    """AgentCard is served at /.well-known/agent-card.json."""
    resp = await client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "Tool Agent"
    assert "skills" in card
    assert len(card["skills"]) >= 2
    assert card["capabilities"]["streaming"] is True


@pytest.mark.asyncio
async def test_agent_card_has_skills_for_tools(client):
    """Each registered tool should appear as an A2A skill."""
    resp = await client.get("/.well-known/agent-card.json")
    card = resp.json()
    skill_ids = {s["id"] for s in card["skills"]}
    assert "http_request" in skill_ids
    assert "json_transform" in skill_ids


@pytest.mark.asyncio
async def test_agent_card_io_modes(client):
    """AgentCard should declare input/output modes."""
    resp = await client.get("/.well-known/agent-card.json")
    card = resp.json()
    assert "text/plain" in card.get("defaultInputModes", [])
    assert "application/json" in card.get("defaultOutputModes", [])


@pytest.mark.asyncio
async def test_a2a_task_execution(client):
    """Submit a task via A2A JSON-RPC and get a result."""
    resp = await client.post("/a2a/", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": "Make an http request to https://example.com"}],
                "messageId": "msg-1",
            },
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


@pytest.mark.asyncio
async def test_a2a_empty_message(client):
    """A2A should handle empty message gracefully."""
    resp = await client.post("/a2a/", json={
        "jsonrpc": "2.0",
        "id": 2,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": ""}],
                "messageId": "msg-2",
            },
        },
    })
    assert resp.status_code == 200
