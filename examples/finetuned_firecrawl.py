#!/usr/bin/env python3
"""Example: Fine-tuned FunctionGemma + Tool Agent + Firecrawl MCP

Architecture:

    User query
      → Fine-tuned FunctionGemma v1 (LoRA adapter, local inference)
        → Tool Agent (routes to best tool)
          → Firecrawl MCP (scrapes/crawls the web)
      → Gemini summarizes results (optional)

This demo validates the fine-tuned model's tool-calling ability
end-to-end through the agent pipeline.

Prerequisites:
    pip install -e ".[dev]"
    export FIRECRAWL_API_KEY="..."
    export GEMINI_API_KEY="..."     # optional, for summarization

Usage:
    python examples/finetuned_firecrawl.py
    python examples/finetuned_firecrawl.py "Scrape https://example.com"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.mcp_client import McpToolBridge
from agent.model import FunctionCall, TransformersBackend
from agent.router import ToolRouter
from agent.tool_registry import ToolRegistry
from agent.tools.base import ToolCategory
from agent.tools.http import register_http_tools

MODEL_PATH = os.getenv(
    "FINETUNED_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "finetuned"),
)

QUERIES = [
    "Scrape https://example.com and get the page content",
    "Search the web for latest AI agent frameworks 2025",
    "Crawl https://news.ycombinator.com and extract the top stories",
]

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Fine-tuned FunctionGemma  →  Tool Agent  →  Firecrawl     ║
╚══════════════════════════════════════════════════════════════╝
"""


async def run_query(
    query: str,
    router: ToolRouter,
    backend: TransformersBackend,
    query_num: int,
    total: int,
):
    """Run a single query through the pipeline and display results."""
    print(f"\n{'='*60}")
    print(f"  Query {query_num}/{total}: \"{query}\"")
    print(f"{'='*60}")

    schemas = router.registry.get_function_schemas()
    print(f"\n  [Model] Generating tool call with {len(schemas)} tools available...")

    t0 = time.perf_counter()
    raw_output = await backend.generate(
        user_message=query,
        tools=schemas,
        max_tokens=256,
        temperature=0.1,
    )
    elapsed = time.perf_counter() - t0

    print(f"  [Model] Raw output ({elapsed:.2f}s):")
    print(f"    {raw_output[:500]}")

    calls = FunctionCall.parse(raw_output)
    print(f"\n  [Parse] Found {len(calls)} function call(s):")
    for i, fc in enumerate(calls, 1):
        print(f"    #{i} {fc.name}({json.dumps(fc.arguments, default=str)[:200]})")

    if not calls:
        print("  [WARN] No function calls parsed — model did not select a tool.")
        return {"query": query, "success": False, "reason": "no_tool_calls", "raw": raw_output}

    results = []
    for fc in calls:
        print(f"\n  [Execute] Running {fc.name}...")
        try:
            result = await router.registry.execute(fc.name, **fc.arguments)
            results.append(result)
            status = "OK" if result.success else "FAILED"
            print(f"  [Execute] {status}")
            if result.error:
                print(f"    Error: {result.error}")
            if result.data:
                data_str = json.dumps(result.data, default=str) if not isinstance(result.data, str) else result.data
                print(f"    Data: {data_str[:500]}{'...' if len(data_str) > 500 else ''}")
        except Exception as e:
            print(f"  [Execute] Exception: {e}")
            results.append(None)

    return {
        "query": query,
        "success": any(r and r.success for r in results),
        "tool_calls": [fc.to_dict() for fc in calls],
        "raw": raw_output,
        "elapsed": elapsed,
    }


async def main():
    firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not firecrawl_key:
        print("Error: FIRECRAWL_API_KEY is not set.")
        sys.exit(1)

    queries = sys.argv[1:] if len(sys.argv) > 1 else QUERIES

    print(BANNER)
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Queries:    {len(queries)}")
    print()

    # ── 1. Load fine-tuned model ─────────────────────────────────
    print("[1/4] Loading fine-tuned FunctionGemma model...")
    t0 = time.perf_counter()
    backend = TransformersBackend(model_path=MODEL_PATH)
    backend._load()
    load_time = time.perf_counter() - t0
    print(f"      Model loaded in {load_time:.1f}s")
    print(f"      FunctionGemma mode: {backend._is_functiongemma}")
    print()

    # ── 2. Set up tools ──────────────────────────────────────────
    print("[2/4] Setting up tool registry + Firecrawl MCP...")
    registry = ToolRegistry()
    register_http_tools(registry)

    bridge = McpToolBridge(
        command="npx",
        args=["-y", "firecrawl-mcp"],
        env={**os.environ, "FIRECRAWL_API_KEY": firecrawl_key},
        category=ToolCategory.INTEGRATION,
        tags=["web", "scraping", "firecrawl", "mcp"],
    )

    async with bridge:
        count = bridge.register_tools(registry)
        print(f"      {count} Firecrawl tools + HTTP tools registered")
        all_tools = registry.list_tools()
        for t in all_tools:
            print(f"        • {t['name']:30s} {t.get('description', '')[:50]}")
        print()

        router = ToolRouter(registry=registry, backend=backend)

        # ── 3. Run queries ───────────────────────────────────────
        print("[3/4] Running queries through fine-tuned model...")

        all_results = []
        for i, q in enumerate(queries, 1):
            result = await run_query(q, router, backend, i, len(queries))
            all_results.append(result)

        # ── 4. Summary ──────────────────────────────────────────
        print(f"\n{'='*60}")
        print("[4/4] Results Summary")
        print(f"{'='*60}")
        success_count = sum(1 for r in all_results if r.get("success"))
        print(f"  Total queries:    {len(all_results)}")
        print(f"  Successful:       {success_count}")
        print(f"  Failed:           {len(all_results) - success_count}")
        print()

        for r in all_results:
            status = "✓" if r.get("success") else "✗"
            calls = r.get("tool_calls", [])
            tool_names = ", ".join(c["name"] for c in calls) if calls else "none"
            elapsed = r.get("elapsed", 0)
            print(f"  {status} [{elapsed:.2f}s] \"{r['query'][:50]}\"")
            print(f"    Tools: {tool_names}")
        print()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
