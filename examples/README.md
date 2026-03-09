# Tool Agent — Examples

Client examples for all four protocols, plus an end-to-end integration demo.

## Protocol Examples

Start the agent with the **mock backend** (no Ollama needed):

```bash
TOOL_AGENT_BACKEND=mock python -m agent.server
```

Then run any of these clients:

| Script | Protocol | What it demonstrates |
|--------|----------|---------------------|
| `rest_client.py` | REST | Health, tool listing, schema, routing, execution |
| `websocket_client.py` | WebSocket | JSON-RPC over WebSocket: list, search, route, execute |
| `mcp_client.py` | MCP | Model Context Protocol: initialize, tools/list, tools/call, resources |
| `a2a_client.py` | A2A | Agent-to-Agent: discover via AgentCard, send tasks |

```bash
python examples/rest_client.py
python examples/websocket_client.py    # needs: pip install websockets
python examples/mcp_client.py
python examples/a2a_client.py
```

---

## Integration Demo: Gemini + Firecrawl MCP

`gemini_firecrawl.py` demonstrates the full architecture — Gemini 2.5 Flash Lite
as the brain, the Tool Agent as the routing layer, and Firecrawl's MCP server
as the web scraping backend.

### Architecture

```
User query
  → Gemini 2.5 Flash Lite    (lightweight model, native function calling)
    → Tool Agent               (routes to best tool from 12+ Firecrawl tools)
      → Firecrawl MCP Server   (scrapes / crawls / searches the web)
  → Gemini summarizes results
```

### Prerequisites

```bash
pip install -e .
export GEMINI_API_KEY="..."        # https://aistudio.google.com/apikey
export FIRECRAWL_API_KEY="..."     # https://firecrawl.dev
```

Node.js is needed for the Firecrawl MCP server (`npx` auto-installs it).

### Run

```bash
python examples/gemini_firecrawl.py
python examples/gemini_firecrawl.py "Scrape https://news.ycombinator.com"
python examples/gemini_firecrawl.py "Search the web for latest AI agent frameworks"
```

### Firecrawl Tools Discovered (12 total)

When the MCP bridge connects, it discovers these tools:

| Tool | Description |
|------|-------------|
| `firecrawl_scrape` | Scrape content from a single URL with advanced options |
| `firecrawl_map` | Map a website to discover all indexed URLs |
| `firecrawl_search` | Search the web and optionally extract content |
| `firecrawl_crawl` | Start a crawl job on a website |
| `firecrawl_check_crawl_status` | Check the status of a crawl job |
| `firecrawl_extract` | Extract structured information from web pages using LLM |
| `firecrawl_agent` | Autonomous web research agent |
| `firecrawl_agent_status` | Check status of an agent job |
| `firecrawl_browser_create` | Create a browser session for code execution |
| `firecrawl_browser_execute` | Execute code in a browser session |
| `firecrawl_browser_delete` | Destroy a browser session |
| `firecrawl_browser_list` | List browser sessions |

Gemini decides which tool to call based on the user's query — no manual routing needed.

---

### Run Results (March 9, 2026)

#### Example 1: Scrape a URL

```
Query: "Scrape https://example.com and summarize the page content"
```

**Gemini selected:** `firecrawl_scrape` with `{"url": "https://example.com", "formats": ["markdown"]}`

**Firecrawl returned:**

```json
{
  "markdown": "# Example Domain\n\nThis domain is for use in documentation examples...",
  "metadata": {
    "title": "Example Domain",
    "language": "en",
    "statusCode": 200,
    "url": "http://example.com/"
  }
}
```

**Gemini summary:**

> This page, "Example Domain," is intended for use in documentation examples.
> It explains that the domain can be used without needing permission, but advises
> against using it in live operations. The page also provides a link to learn more
> about example domains.

#### Example 2: Web Search

```
Query: "Search the web for latest AI agent frameworks in 2026"
```

**Gemini selected:** `firecrawl_search` with `{"query": "latest AI agent frameworks in 2026"}`

**Firecrawl returned:** 5 web results including:

| # | Title | Source |
|---|-------|--------|
| 1 | 12 Best AI Agent Frameworks in 2026 | medium.com |
| 2 | Top tools to build AI agents in 2026 | reddit.com |
| 3 | The 4 Best Open Source Multi-Agent AI Frameworks 2026 | towardsai.net |

**Gemini summary:**

> The search reveals several prominent options. LangGraph and CrewAI are
> frequently mentioned as leading choices, with LangGraph highlighted for
> its graph-native, stateful architecture and persistent memory capabilities.
> AutoGen is also a strong contender, offering a versatile framework with tools
> for both coders and non-coders. Other notable frameworks include Semantic
> Kernel, Pydantic AI, and LangChain.

#### Example 3: Scrape Hacker News

```
Query: "Scrape https://news.ycombinator.com and tell me the top stories"
```

**Gemini selected:** `firecrawl_scrape` with `{"formats": ["markdown"], "url": "https://news.ycombinator.com"}`

**Gemini summary:**

> Top stories from Hacker News:
> 1. **Restoring a Sun SPARCstation IPX Part 1: PSU and Nvram** — restoration of vintage hardware.
> 2. **Is legal the same as legitimate: AI reimplementation and the erosion of copyleft** — legal and ethical implications of AI in software.
> 3. **Fontcrafter: Turn Your Handwriting into a Real Font** — a tool for creating custom fonts.
> 4. **Flash media longevity testing – 6 years later** — findings on flash storage durability.
> 5. **Ireland shuts last coal plant, becomes 15th coal-free country in Europe** — renewable energy transition.

---

### Key Observations

- **Gemini selects the right tool automatically** — `firecrawl_scrape` for URLs, `firecrawl_search` for web search queries
- **End-to-end latency is ~3-8 seconds** (MCP subprocess startup + Gemini API + Firecrawl API)
- **12 Firecrawl tools are available** but Gemini only picks what's relevant to the query
- **The model correctly populates arguments** — infers `formats: ["markdown"]` for scraping, constructs search queries from natural language
- **No manual tool routing needed** — the lightweight Gemini model handles function calling natively

### Use a different Gemini model

```bash
GEMINI_MODEL=gemini-2.5-flash python examples/gemini_firecrawl.py
```

### Run the full server with Gemini

```bash
TOOL_AGENT_BACKEND=gemini GEMINI_API_KEY="..." python -m agent.server
```

All protocols (REST, WebSocket, A2A, MCP) will use Gemini for tool selection.

---

## Fine-tuned FunctionGemma v1 — End-to-End Evaluation

`finetuned_firecrawl.py` and `finetuned_simple_test.py` test the fine-tuned
FunctionGemma 270M model (LoRA adapter merged at runtime) through the full
tool_agent pipeline. `base_vs_finetuned_test.py` runs both models side-by-side.

### What Changed to Make It Work

The `TransformersBackend` was updated to:

1. **Auto-detect LoRA adapters** — reads `adapter_config.json`, loads the base model, and merges the adapter
2. **Detect FunctionGemma models** — uses Gemma 3 turn markers (`<start_of_turn>user/model`) instead of chat template
3. **Use legacy system prompt** — matches the training format: `"You are a model that can do function calling with the following functions"`

`FunctionCall.parse()` already handled both JSON and legacy `<start_function_call>` formats.

### Run

```bash
# Simple tool calling test (no API keys needed)
python examples/finetuned_simple_test.py

# Firecrawl MCP integration
FIRECRAWL_API_KEY="..." python examples/finetuned_firecrawl.py

# Side-by-side comparison (downloads base model on first run)
python examples/base_vs_finetuned_test.py
```

### Results: Base vs Fine-tuned (March 9, 2026)

| Metric | Base (270M) | Fine-tuned v1 | Delta |
|--------|------------|---------------|-------|
| Tool Selection Accuracy | 1/7 (14%) | 4/7 (57%) | **+43%** |
| Avg inference time | 2.21s | 2.72s | +0.51s |

Per-query breakdown:

| Query | Base | Fine-tuned |
|-------|------|-----------|
| "What's the weather in Tokyo?" | ✗ (none) | ✓ `get_weather` |
| "Search for latest news about AI" | ✗ (none) | ✓ `search_web` |
| "Send email to john@example.com..." | ✗ (none) | ✓ `send_email` |
| "What is 234 * 567 + 89?" | ✗ (none) | ✗ `search_web` (expected: `calculate`) |
| "Remind me to call the dentist at 9am" | ✗ (none) | ✗ `send_email` (expected: `set_reminder`) |
| "Weather in Paris in fahrenheit?" | ✗ (none) | ✓ `get_weather` |
| "Tell me a joke about programming" | ✓ (none) | ✗ `search_web` (expected: none) |

### Fine-tuned + Firecrawl MCP

When tested with the full 14-tool registry (HTTP + 12 Firecrawl tools), the
270M model struggles with complex, multi-parameter Firecrawl schemas it has
never seen during training:

- **Query 1** ("Scrape https://example.com"): Selected `firecrawl_browser_create` instead of `firecrawl_scrape`, passed wrong types
- **Query 2** ("Search the web for AI frameworks"): No tool selected
- **Query 3** ("Crawl HN and extract top stories"): No tool selected

### Key Observations

- **Fine-tuning works**: The base model produced zero valid tool calls (14% accuracy = 1 lucky "no call" match). Fine-tuned v1 correctly selects tools for 57% of queries
- **Format is correct**: The model generates proper `<start_function_call>call:fn{key:<escape>val<escape>}` tokens, parsed successfully by `FunctionCall.parse()`
- **270M is too small for complex schemas**: With 14+ tools and parameters with nested types, the model defaults to familiar tools (`search_web`, `send_email`) or gives up
- **Argument extraction is strong**: When it picks the right tool, arguments are extracted correctly (city names, email addresses, subjects)
- **Negligible speed overhead**: LoRA merge adds ~0.5s to load time, no inference cost
