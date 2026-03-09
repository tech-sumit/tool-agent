"""Integration tests for WebSocket JSON-RPC protocol."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from agent.server import app


class TestWebSocket:
    def setup_method(self):
        self.client = TestClient(app)

    def test_connect_and_welcome(self):
        with self.client.websocket_connect("/ws") as ws:
            welcome = ws.receive_json()
            assert welcome["method"] == "connected"
            assert "session_id" in welcome["params"]
            assert welcome["params"]["tools_available"] >= 2

    def test_list_tools(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome
            ws.send_json({
                "jsonrpc": "2.0",
                "id": "1",
                "method": "list_tools",
                "params": {},
            })
            resp = ws.receive_json()
            assert resp["id"] == "1"
            assert resp["result"]["total"] >= 2
            names = {t["name"] for t in resp["result"]["tools"]}
            assert "http_request" in names

    def test_get_schema(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "jsonrpc": "2.0",
                "id": "2",
                "method": "get_schema",
                "params": {"tool": "http_request"},
            })
            resp = ws.receive_json()
            assert resp["id"] == "2"
            assert resp["result"]["type"] == "function"
            assert resp["result"]["function"]["name"] == "http_request"

    def test_search_tools(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "jsonrpc": "2.0",
                "id": "3",
                "method": "search_tools",
                "params": {"query": "json"},
            })
            resp = ws.receive_json()
            assert resp["id"] == "3"
            assert resp["result"]["total"] >= 1

    def test_route(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "jsonrpc": "2.0",
                "id": "4",
                "method": "route",
                "params": {"message": "Make an http request to example.com", "execute": False},
            })
            resp = ws.receive_json()
            assert resp["id"] == "4"
            assert "function_calls" in resp["result"]
            assert len(resp["result"]["function_calls"]) >= 1

    def test_execute(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "jsonrpc": "2.0",
                "id": "5",
                "method": "execute",
                "params": {
                    "tool": "json_transform",
                    "parameters": {
                        "data": '{"key": "value"}',
                        "expression": "key",
                    },
                },
            })
            resp = ws.receive_json()
            assert resp["id"] == "5"
            assert resp["result"]["success"] is True
            assert resp["result"]["data"] == "value"

    def test_unknown_method(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({
                "jsonrpc": "2.0",
                "id": "6",
                "method": "nonexistent",
                "params": {},
            })
            resp = ws.receive_json()
            assert resp["id"] == "6"
            assert "error" in resp
            assert resp["error"]["code"] == -32601

    def test_invalid_json(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_text("not json at all")
            resp = ws.receive_json()
            assert "error" in resp
            assert resp["error"]["code"] == -32700

    def test_multiple_requests(self):
        with self.client.websocket_connect("/ws") as ws:
            ws.receive_json()
            for i in range(3):
                ws.send_json({
                    "jsonrpc": "2.0",
                    "id": f"multi-{i}",
                    "method": "list_tools",
                    "params": {},
                })
                resp = ws.receive_json()
                assert resp["id"] == f"multi-{i}"
                assert resp["result"]["total"] >= 2
