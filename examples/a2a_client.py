#!/usr/bin/env python3
"""Example: A2A (Agent-to-Agent) client for Tool Agent.

Demonstrates the Google A2A protocol:
  - Discover the agent via /.well-known/agent-card.json
  - Send a task via JSON-RPC message/send
  - Inspect the task result

Usage:
    # Start the agent first:
    TOOL_AGENT_BACKEND=mock python -m agent.server

    # Then run this script:
    python examples/a2a_client.py

Requires: httpx (`pip install httpx`)
"""

import json
import uuid

import httpx

BASE = "http://localhost:8888"


def main():
    with httpx.Client(base_url=BASE, timeout=30) as client:

        # 1. Discover agent
        print("=" * 60)
        print("  1. Discover Agent (A2A AgentCard)")
        print("=" * 60)
        resp = client.get("/.well-known/agent-card.json")
        resp.raise_for_status()
        card = resp.json()
        print(f"  Name:         {card['name']}")
        print(f"  Description:  {card.get('description', '')[:70]}")
        print(f"  Version:      {card.get('version', '?')}")
        print(f"  Streaming:    {card['capabilities'].get('streaming', False)}")
        print(f"  Skills:       {len(card['skills'])}")
        for skill in card["skills"][:5]:
            print(f"    - {skill['name']:25s} {skill.get('description', '')[:45]}")
        if len(card["skills"]) > 5:
            print(f"    ... and {len(card['skills']) - 5} more")
        print()

        # 2. Send a task
        print("=" * 60)
        print("  2. Send Task via A2A")
        print("=" * 60)
        task_message = "Make an http request to https://httpbin.org/get"
        print(f"  Message: {task_message}")
        print()

        resp = client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"text": task_message}],
                    "messageId": str(uuid.uuid4()),
                },
            },
        })
        resp.raise_for_status()
        data = resp.json()

        if "result" in data:
            result = data["result"]
            status = result.get("status", {})
            state = status.get("state", "?")
            print(f"  Task ID:  {result.get('id', '?')}")
            print(f"  State:    {state}")

            message = status.get("message", {})
            if message:
                parts = message.get("parts", [])
                for part in parts:
                    if hasattr(part, "get"):
                        text = part.get("text", part.get("root", {}).get("text", ""))
                    else:
                        text = str(part)
                    if text:
                        try:
                            parsed = json.loads(text)
                            print(f"  Result:")
                            for fc in parsed.get("function_calls", []):
                                print(f"    Tool:      {fc['name']}")
                                print(f"    Arguments: {fc['arguments']}")
                        except (json.JSONDecodeError, TypeError):
                            print(f"  Response: {text[:200]}")
        elif "error" in data:
            print(f"  Error: {data['error']}")
        print()

        # 3. Send another task
        print("=" * 60)
        print("  3. Another Task")
        print("=" * 60)
        resp = client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"text": "Transform json data to extract a field"}],
                    "messageId": str(uuid.uuid4()),
                },
            },
        })
        data = resp.json()
        if "result" in data:
            state = data["result"].get("status", {}).get("state", "?")
            print(f"  State: {state}")
        print()


if __name__ == "__main__":
    main()
