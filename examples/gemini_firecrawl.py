#!/usr/bin/env python3
"""Example: Gemini Flash Lite + Tool Agent + Firecrawl MCP

Architecture:

    User query
      → Gemini 2.0 Flash Lite  (lightweight model, native function calling)
        → Tool Agent            (routes to best tool)
          → Firecrawl MCP       (scrapes/crawls the web)
      → Gemini summarizes results

This demo shows:
  1. Spawning Firecrawl's MCP server as a subprocess
  2. Bridging its tools into the Tool Agent's registry
  3. Gemini deciding which Firecrawl tool to call, with what arguments
  4. Executing the tool via the MCP bridge
  5. Feeding results back to Gemini for summarization

Prerequisites:
    pip install -e .                    # install the agent
    npm install -g firecrawl-mcp       # or rely on npx (auto-installs)
    export GEMINI_API_KEY="..."        # Google AI Studio key
    export FIRECRAWL_API_KEY="..."     # firecrawl.dev key

Usage:
    python examples/gemini_firecrawl.py
    python examples/gemini_firecrawl.py "Scrape https://news.ycombinator.com"
    python examples/gemini_firecrawl.py "Search the web for latest AI agent frameworks"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.mcp_client import McpToolBridge
from agent.model import GeminiBackend
from agent.router import ToolRouter
from agent.tool_registry import ToolRegistry
from agent.tools.base import ToolCategory
from agent.tools.http import register_http_tools

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          Gemini  →  Tool Agent  →  Firecrawl MCP            ║
╚══════════════════════════════════════════════════════════════╝
"""


async def main():
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "")

    if not gemini_key:
        print("Error: GEMINI_API_KEY is not set.")
        print("  Get one at https://aistudio.google.com/apikey")
        sys.exit(1)
    if not firecrawl_key:
        print("Error: FIRECRAWL_API_KEY is not set.")
        print("  Get one at https://firecrawl.dev")
        sys.exit(1)

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Scrape https://example.com and summarize the page content"
    )

    print(BANNER)

    registry = ToolRegistry()
    register_http_tools(registry)

    # ── 1. Connect to Firecrawl MCP server ──────────────────────────

    print("[1/5] Starting Firecrawl MCP server...")
    bridge = McpToolBridge(
        command="npx",
        args=["-y", "firecrawl-mcp"],
        env={**os.environ, "FIRECRAWL_API_KEY": firecrawl_key},
        category=ToolCategory.INTEGRATION,
        tags=["web", "scraping", "firecrawl", "mcp"],
    )

    async with bridge:
        count = bridge.register_tools(registry)
        print(f"      {count} Firecrawl tools registered:")
        for tool in bridge.tools:
            print(f"        • {tool.name:30s} {tool.description[:50]}")
        print()

        # ── 2. Initialize Gemini backend ────────────────────────────

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        print(f"[2/5] Initializing Gemini backend ({model_name})...")
        backend = GeminiBackend(model_name=model_name, api_key=gemini_key)
        healthy = await backend.health_check()
        print(f"      Model healthy: {healthy}")
        if not healthy:
            print("      Error: Cannot reach Gemini API. Check your API key.")
            return
        print()

        # ── 3. Route the query ──────────────────────────────────────

        router = ToolRouter(registry=registry, backend=backend)

        print(f"[3/5] Query: \"{query}\"")
        print("      Asking Gemini to select tools...")
        print()

        result = await router.route(message=query, execute=True)

        # ── 4. Show results ─────────────────────────────────────────

        print(f"[4/5] Execution results:")
        print(f"      Success: {result.success}")

        if result.function_calls:
            for i, fc in enumerate(result.function_calls, 1):
                print(f"      Tool #{i}: {fc.name}")
                args_str = json.dumps(fc.arguments, indent=8, default=str)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                print(f"      Args:   {args_str}")
        else:
            print(f"      Raw output: {result.raw_output[:300]}")

        if result.results:
            for i, r in enumerate(result.results, 1):
                print(f"\n      Result #{i}: {'OK' if r.success else 'FAILED'}")
                if r.error:
                    print(f"      Error: {r.error}")
                if r.data:
                    data_str = (
                        json.dumps(r.data, indent=8, default=str)
                        if not isinstance(r.data, str)
                        else r.data
                    )
                    if len(data_str) > 1500:
                        data_str = data_str[:1500] + "\n      ... (truncated)"
                    print(f"      Data:\n{data_str}")

        if result.error:
            print(f"\n      Error: {result.error}")
        print()

        # ── 5. Summarize with Gemini ────────────────────────────────

        if result.success and result.results:
            print("[5/5] Asking Gemini to summarize the scraped content...")
            print()

            scraped = ""
            for r in result.results:
                if r.data:
                    chunk = (
                        json.dumps(r.data, default=str)
                        if not isinstance(r.data, str)
                        else r.data
                    )
                    scraped += chunk[:4000]

            summary = await backend.generate(
                user_message=(
                    f"I scraped a webpage with this query: \"{query}\"\n\n"
                    f"Here is the scraped content:\n\n{scraped}\n\n"
                    "Give a concise, informative summary (3-5 sentences)."
                ),
                tools=[],
                max_tokens=500,
                temperature=0.3,
            )
            print("      ┌─ Summary ──────────────────────────────────────")
            for line in summary.strip().split("\n"):
                print(f"      │ {line}")
            print("      └───────────────────────────────────────────────")
        else:
            print("[5/5] Skipped summarization (no successful results).")

        await backend.close()

    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
