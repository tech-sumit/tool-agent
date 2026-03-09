#!/usr/bin/env python3
"""Example: WebSocket JSON-RPC client for Tool Agent.

Demonstrates real-time interaction via WebSocket:
  - Connect and receive welcome message
  - list_tools  — enumerate registered tools
  - get_schema  — get a tool's JSON schema
  - route       — model-powered tool selection
  - execute     — direct tool execution

Usage:
    # Start the agent first:
    TOOL_AGENT_BACKEND=mock python -m agent.server

    # Then run this script:
    python examples/websocket_client.py

Requires: websockets (`pip install websockets`)
"""

import asyncio
import json

import websockets


async def main():
    uri = "ws://localhost:8888/ws"
    req_id = 0

    async def call(ws, method: str, params: dict) -> dict:
        nonlocal req_id
        req_id += 1
        msg = {"jsonrpc": "2.0", "id": str(req_id), "method": method, "params": params}
        await ws.send(json.dumps(msg))
        resp = json.loads(await ws.recv())
        return resp

    async with websockets.connect(uri) as ws:
        # Welcome message
        welcome = json.loads(await ws.recv())
        print("=" * 60)
        print("  Connected to Tool Agent via WebSocket")
        print("=" * 60)
        print(f"  Session ID:       {welcome['params']['session_id']}")
        print(f"  Tools available:  {welcome['params']['tools_available']}")
        print()

        # 1. List tools
        print("=" * 60)
        print("  1. list_tools")
        print("=" * 60)
        resp = await call(ws, "list_tools", {})
        for tool in resp["result"]["tools"]:
            print(f"  {tool['name']:20s} [{tool['category']}]")
        print()

        # 2. Get schema
        print("=" * 60)
        print("  2. get_schema (http_request)")
        print("=" * 60)
        resp = await call(ws, "get_schema", {"tool": "http_request"})
        func = resp["result"]["function"]
        print(f"  Name:       {func['name']}")
        print(f"  Parameters: {list(func['parameters']['properties'].keys())}")
        print()

        # 3. Search tools
        print("=" * 60)
        print("  3. search_tools (query='json')")
        print("=" * 60)
        resp = await call(ws, "search_tools", {"query": "json"})
        for r in resp["result"]["results"]:
            print(f"  {r['name']:20s} {r['description'][:50]}")
        print()

        # 4. Route a message
        print("=" * 60)
        print("  4. route")
        print("=" * 60)
        resp = await call(ws, "route", {
            "message": "Make an http request to https://httpbin.org/get",
            "execute": False,
        })
        result = resp["result"]
        for fc in result["function_calls"]:
            print(f"  Selected: {fc['name']}")
            print(f"  Args:     {fc['arguments']}")
        print()

        # 5. Execute a tool
        print("=" * 60)
        print("  5. execute (json_transform)")
        print("=" * 60)
        resp = await call(ws, "execute", {
            "tool": "json_transform",
            "parameters": {
                "data": '{"name": "Tool Agent", "version": "0.1.0"}',
                "expression": "name",
            },
        })
        print(f"  Success: {resp['result']['success']}")
        print(f"  Data:    {resp['result']['data']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
