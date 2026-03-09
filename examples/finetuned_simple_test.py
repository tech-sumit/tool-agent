#!/usr/bin/env python3
"""Test: Fine-tuned FunctionGemma v1 — simple tool calling evaluation.

Tests the model with simple, well-defined tools that are closer to its
training distribution (Glaive/ToolBench style).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.model import FunctionCall, TransformersBackend, LEGACY_SYSTEM_PROMPT
from agent.tools.base import ToolCategory, ToolParameter, ToolSchema

MODEL_PATH = os.getenv(
    "FINETUNED_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "finetuned"),
)

SIMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body content"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder for a specific time",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Reminder message"},
                "time": {"type": "string", "description": "Time for the reminder (ISO 8601)"},
            },
            "required": ["message", "time"],
        },
    },
]

TEST_CASES = [
    {
        "query": "What's the weather in Tokyo?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "Tokyo"},
    },
    {
        "query": "Search for the latest news about artificial intelligence",
        "expected_tool": "search_web",
        "expected_args": {"query": "latest news about artificial intelligence"},
    },
    {
        "query": "Send an email to john@example.com with subject 'Meeting' and body 'See you at 3pm'",
        "expected_tool": "send_email",
        "expected_args": {"to": "john@example.com", "subject": "Meeting"},
    },
    {
        "query": "What is 234 * 567 + 89?",
        "expected_tool": "calculate",
        "expected_args": {"expression": "234 * 567 + 89"},
    },
    {
        "query": "Remind me to call the dentist tomorrow at 9am",
        "expected_tool": "set_reminder",
        "expected_args": {"message": "call the dentist"},
    },
    {
        "query": "What's the weather in Paris in fahrenheit?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "Paris", "unit": "fahrenheit"},
    },
    {
        "query": "Tell me a joke about programming",
        "expected_tool": None,
        "expected_args": {},
    },
]


async def main():
    print("=" * 60)
    print("  Fine-tuned FunctionGemma v1 — Simple Tool Calling Test")
    print("=" * 60)
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Test cases: {len(TEST_CASES)}")
    print()

    print("[1/3] Loading model...")
    t0 = time.perf_counter()
    backend = TransformersBackend(model_path=MODEL_PATH)
    backend._load()
    print(f"      Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"      FunctionGemma: {backend._is_functiongemma}")
    print()

    print("[2/3] Running test cases...")
    print()

    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        query = tc["query"]
        expected_tool = tc["expected_tool"]
        expected_args = tc["expected_args"]

        print(f"  Test {i}/{len(TEST_CASES)}: \"{query}\"")
        print(f"    Expected: {expected_tool or '(no tool call)'}")

        t0 = time.perf_counter()
        raw = await backend.generate(
            user_message=query,
            tools=SIMPLE_TOOLS,
            max_tokens=256,
            temperature=0.1,
        )
        elapsed = time.perf_counter() - t0

        calls = FunctionCall.parse(raw)
        actual_tool = calls[0].name if calls else None
        actual_args = calls[0].arguments if calls else {}

        tool_match = actual_tool == expected_tool
        arg_match = all(
            k in actual_args and (expected_args[k].lower() in str(actual_args[k]).lower())
            for k in expected_args
        ) if expected_args else True

        status = "PASS" if tool_match else "FAIL"
        arg_status = "args_ok" if arg_match else "args_mismatch"

        print(f"    Actual:   {actual_tool or '(no tool call)'}")
        if actual_args:
            print(f"    Args:     {json.dumps(actual_args, default=str)[:200]}")
        print(f"    Raw:      {raw[:200]}")
        print(f"    Result:   {status} | {arg_status} | {elapsed:.2f}s")
        print()

        results.append({
            "query": query,
            "expected_tool": expected_tool,
            "actual_tool": actual_tool,
            "tool_match": tool_match,
            "arg_match": arg_match,
            "expected_args": expected_args,
            "actual_args": actual_args,
            "raw": raw,
            "elapsed": elapsed,
        })

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("[3/3] Summary")
    print("=" * 60)

    tool_correct = sum(1 for r in results if r["tool_match"])
    arg_correct = sum(1 for r in results if r["tool_match"] and r["arg_match"])
    total = len(results)

    print(f"  Tool Selection Accuracy:  {tool_correct}/{total} ({100*tool_correct/total:.0f}%)")
    print(f"  Full Match (tool + args): {arg_correct}/{total} ({100*arg_correct/total:.0f}%)")
    print()

    for r in results:
        icon = "✓" if r["tool_match"] else "✗"
        arg_icon = "✓" if r["arg_match"] else "~"
        print(f"  {icon}{arg_icon} [{r['elapsed']:.2f}s] \"{r['query'][:45]}\"")
        if not r["tool_match"]:
            print(f"     expected={r['expected_tool']}, got={r['actual_tool']}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
