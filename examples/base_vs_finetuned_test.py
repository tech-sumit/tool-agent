#!/usr/bin/env python3
"""Comparison: Base FunctionGemma vs Fine-tuned v1 through the tool_agent pipeline.

Loads both models and runs identical queries to measure the
improvement from fine-tuning.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.model import FunctionCall, TransformersBackend

FINETUNED_PATH = os.getenv(
    "FINETUNED_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "finetuned"),
)
BASE_MODEL = "unsloth/functiongemma-270m-it"

SIMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
                "num_results": {"type": "integer", "description": "Number of results"},
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
                "to": {"type": "string", "description": "Recipient email"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
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
                "expression": {"type": "string", "description": "Math expression to evaluate"},
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
                "time": {"type": "string", "description": "Time for the reminder"},
            },
            "required": ["message", "time"],
        },
    },
]

TEST_CASES = [
    ("What's the weather in Tokyo?", "get_weather"),
    ("Search for latest news about artificial intelligence", "search_web"),
    ("Send an email to john@example.com with subject 'Meeting' and body 'See you at 3pm'", "send_email"),
    ("What is 234 * 567 + 89?", "calculate"),
    ("Remind me to call the dentist tomorrow at 9am", "set_reminder"),
    ("What's the weather in Paris in fahrenheit?", "get_weather"),
    ("Tell me a joke about programming", None),
]


async def evaluate_model(backend: TransformersBackend, label: str) -> list[dict]:
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {label}")
    print(f"{'─'*60}")

    results = []
    for i, (query, expected) in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        raw = await backend.generate(
            user_message=query, tools=SIMPLE_TOOLS, max_tokens=256, temperature=0.1,
        )
        elapsed = time.perf_counter() - t0

        calls = FunctionCall.parse(raw)
        actual = calls[0].name if calls else None
        match = actual == expected

        icon = "✓" if match else "✗"
        print(f"  {icon} [{elapsed:.1f}s] \"{query[:50]}\" → {actual or '(none)'} (expected: {expected or '(none)'})")

        results.append({"query": query, "expected": expected, "actual": actual, "match": match, "elapsed": elapsed, "raw": raw[:300]})

    return results


async def main():
    print("=" * 60)
    print("  Base vs Fine-tuned FunctionGemma — tool_agent Pipeline")
    print("=" * 60)

    # Load fine-tuned
    print("\n[1/3] Loading fine-tuned model...")
    ft_backend = TransformersBackend(model_path=FINETUNED_PATH)
    ft_backend._load()
    print(f"      Done. FunctionGemma={ft_backend._is_functiongemma}")

    # Load base
    print("\n[2/3] Loading base model...")
    base_backend = TransformersBackend(model_path=BASE_MODEL)
    base_backend._load()
    print(f"      Done. FunctionGemma={base_backend._is_functiongemma}")

    # Evaluate both
    ft_results = await evaluate_model(ft_backend, "Fine-tuned v1")
    base_results = await evaluate_model(base_backend, "Base (unsloth/functiongemma-270m-it)")

    # Comparison
    print(f"\n{'='*60}")
    print("[3/3] Comparison")
    print(f"{'='*60}")

    ft_correct = sum(1 for r in ft_results if r["match"])
    base_correct = sum(1 for r in base_results if r["match"])
    total = len(TEST_CASES)

    print(f"\n  {'Metric':<30s} {'Base':>10s} {'Fine-tuned':>12s} {'Delta':>8s}")
    print(f"  {'─'*62}")
    print(f"  {'Tool Selection Accuracy':<30s} {base_correct}/{total} ({100*base_correct/total:.0f}%){' ':>3s} {ft_correct}/{total} ({100*ft_correct/total:.0f}%){' ':>3s} {'+' if ft_correct>=base_correct else ''}{ft_correct-base_correct:+d}")

    ft_avg = sum(r["elapsed"] for r in ft_results) / total
    base_avg = sum(r["elapsed"] for r in base_results) / total
    print(f"  {'Avg inference time':<30s} {base_avg:>8.2f}s   {ft_avg:>10.2f}s   {ft_avg-base_avg:>+.2f}s")

    print(f"\n  Per-query comparison:")
    print(f"  {'Query':<45s} {'Base':>8s} {'FT':>8s}")
    print(f"  {'─'*65}")
    for b, f in zip(base_results, ft_results):
        b_icon = "✓" if b["match"] else "✗"
        f_icon = "✓" if f["match"] else "✗"
        q = b["query"][:43]
        print(f"  {q:<45s} {b_icon} {b['actual'] or '(none)':>8s} {f_icon} {f['actual'] or '(none)':>8s}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
