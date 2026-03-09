#!/usr/bin/env python3
"""Example: REST API client for Tool Agent.

Demonstrates all REST endpoints:
  - GET  /health         — check agent status
  - GET  /tools          — list registered tools
  - GET  /tools/{name}   — get a tool's JSON schema
  - POST /route          — route a natural-language request to tools
  - POST /execute        — execute a specific tool directly

Usage:
    # Start the agent first (mock backend for quick demo):
    TOOL_AGENT_BACKEND=mock python -m agent.server

    # Then run this script:
    python examples/rest_client.py
"""

import httpx

BASE = "http://localhost:8888"


def main():
    with httpx.Client(base_url=BASE, timeout=30) as client:

        # 1. Health check
        print("=" * 60)
        print("  1. Health Check")
        print("=" * 60)
        resp = client.get("/health")
        health = resp.json()
        print(f"  Status:          {health['status']}")
        print(f"  Tools registered: {health['tools_registered']}")
        print(f"  Model healthy:   {health['model_healthy']}")
        print(f"  Protocols:       {', '.join(health['protocols'])}")
        print()

        # 2. List all tools
        print("=" * 60)
        print("  2. List Tools")
        print("=" * 60)
        resp = client.get("/tools")
        tools = resp.json()
        for tool in tools["tools"]:
            print(f"  {tool['name']:20s} [{tool['category']}] {tool['description'][:60]}")
        print(f"  Total: {tools['total']}")
        print()

        # 3. Get tool schema
        print("=" * 60)
        print("  3. Get Tool Schema (http_request)")
        print("=" * 60)
        resp = client.get("/tools/http_request")
        schema = resp.json()
        func = schema["function"]
        print(f"  Name:       {func['name']}")
        print(f"  Parameters: {list(func['parameters']['properties'].keys())}")
        print()

        # 4. Route a message (model selects & calls tools)
        print("=" * 60)
        print("  4. Route Message")
        print("=" * 60)
        resp = client.post("/route", json={
            "message": "Make an http request to https://httpbin.org/get",
            "execute": False,
        })
        result = resp.json()
        print(f"  Success:    {result['success']}")
        for fc in result["function_calls"]:
            print(f"  Tool:       {fc['name']}")
            print(f"  Arguments:  {fc['arguments']}")
        print()

        # 5. Execute a tool directly
        print("=" * 60)
        print("  5. Execute Tool (json_transform)")
        print("=" * 60)
        resp = client.post("/execute", json={
            "tool": "json_transform",
            "parameters": {
                "data": '{"users": [{"name": "Alice"}, {"name": "Bob"}]}',
                "expression": "users",
            },
        })
        result = resp.json()
        print(f"  Success: {result['success']}")
        print(f"  Data:    {result['data']}")
        print()


if __name__ == "__main__":
    main()
