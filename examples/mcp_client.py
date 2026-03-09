#!/usr/bin/env python3
"""Example: MCP (Model Context Protocol) client for Tool Agent.

Demonstrates connecting to the MCP server at /mcp:
  - Initialize session
  - List available MCP tools
  - Call tools (list_tools, route_request, execute_tool)
  - Read resources (tools://catalog, tools://capabilities)

Usage:
    # Start the agent first:
    TOOL_AGENT_BACKEND=mock python -m agent.server

    # Then run this script:
    python examples/mcp_client.py

Requires: httpx (`pip install httpx`)
"""

import json

import httpx

BASE = "http://localhost:8888/mcp/mcp"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def rpc(client: httpx.Client, method: str, params: dict | None = None, req_id: int | None = None) -> dict | None:
    """Send a JSON-RPC request to the MCP server."""
    body: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        body["params"] = params
    if req_id is not None:
        body["id"] = req_id
    resp = client.post(BASE, json=body, headers=HEADERS)
    resp.raise_for_status()
    if req_id is not None:
        return resp.json()
    return None


def main():
    with httpx.Client(timeout=30) as client:

        # 1. Initialize MCP session
        print("=" * 60)
        print("  1. Initialize MCP Session")
        print("=" * 60)
        init = rpc(client, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "example-client", "version": "0.1"},
        }, req_id=1)

        session_id = None
        if init:
            server_info = init.get("result", {}).get("serverInfo", {})
            print(f"  Server: {server_info.get('name', '?')} v{server_info.get('version', '?')}")
            print(f"  Protocol: {init['result'].get('protocolVersion', '?')}")
        print()

        rpc(client, "notifications/initialized")

        # 2. List available MCP tools
        print("=" * 60)
        print("  2. List MCP Tools")
        print("=" * 60)
        tools_resp = rpc(client, "tools/list", {}, req_id=2)
        if tools_resp and "result" in tools_resp:
            for tool in tools_resp["result"]["tools"]:
                desc = tool.get("description", "")[:55]
                print(f"  {tool['name']:25s} {desc}")
        print()

        # 3. Call list_tools (meta-tool that lists agent tools)
        print("=" * 60)
        print("  3. Call: list_tools")
        print("=" * 60)
        resp = rpc(client, "tools/call", {
            "name": "list_tools",
            "arguments": {"category": ""},
        }, req_id=3)
        if resp and "result" in resp:
            tools = json.loads(resp["result"]["content"][0]["text"])
            for t in tools:
                print(f"  {t['name']:20s} [{t['category']}] params={t['parameters']}")
        print()

        # 4. Call route_request (model selects tools)
        print("=" * 60)
        print("  4. Call: route_request")
        print("=" * 60)
        resp = rpc(client, "tools/call", {
            "name": "route_request",
            "arguments": {"message": "Make an http request to https://httpbin.org/get"},
        }, req_id=4)
        if resp and "result" in resp:
            result = json.loads(resp["result"]["content"][0]["text"])
            for fc in result.get("function_calls", []):
                print(f"  Tool:      {fc['name']}")
                print(f"  Arguments: {fc['arguments']}")
        print()

        # 5. Call execute_tool directly
        print("=" * 60)
        print("  5. Call: execute_tool (json_transform)")
        print("=" * 60)
        resp = rpc(client, "tools/call", {
            "name": "execute_tool",
            "arguments": {
                "tool_name": "json_transform",
                "parameters": json.dumps({
                    "data": '{"items": ["a", "b", "c"]}',
                    "expression": "items",
                }),
            },
        }, req_id=5)
        if resp and "result" in resp:
            result = json.loads(resp["result"]["content"][0]["text"])
            print(f"  Success: {result['success']}")
            print(f"  Data:    {result['data']}")
        print()

        # 6. Read resources
        print("=" * 60)
        print("  6. Read Resource: tools://capabilities")
        print("=" * 60)
        resp = rpc(client, "resources/list", {}, req_id=6)
        if resp and "result" in resp:
            for r in resp["result"].get("resources", []):
                print(f"  {r['uri']:30s} {r.get('name', '')}")
        print()


if __name__ == "__main__":
    main()
