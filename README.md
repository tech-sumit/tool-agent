# Tool Agent — FunctionGemma Integration Expert

A fine-tuned FunctionGemma 270M agent that acts as a function/integration expert. Exposes **A2A**, **MCP**, and **WebSocket** protocols so other systems and agents can discover its capabilities and execute tool calls.

Built on Google's [FunctionGemma](https://huggingface.co/google/functiongemma-270m-it) (Gemma 3 270M), fine-tuned with LoRA on 13,000 general function-calling examples from Salesforce xlam-60k and irrelevance/refusal data.

**Model on Hugging Face**: [sumitagrawal/functiongemma-270m-tool-agent](https://huggingface.co/sumitagrawal/functiongemma-270m-tool-agent)

## Benchmark Results

Evaluated using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

| Metric | Base | Fine-tuned | Delta |
|--------|------|-----------|-------|
| Tool Selection Acc | 49.0% | 78.0% | **+29.0%** |
| First Tool Acc | 49.0% | 88.0% | **+39.0%** |
| Negative Rejection | 100.0% | 100.0% | +0.0% |
| Param Accuracy | 49.0% | 68.9% | **+19.9%** |

## Quick Start

```bash
# Install
pip install -e ".[training]"

# Pull the base model into Ollama (or use fine-tuned)
ollama create tool-agent -f models/gguf/Modelfile

# Start the agent
make serve
```

## Protocols

| Protocol | Endpoint | Purpose |
|----------|----------|---------|
| A2A | `/.well-known/agent-card.json` + `/a2a` | Agent-to-Agent discovery & task execution |
| MCP | `/mcp` | Model Context Protocol tool exposure |
| WebSocket | `/ws` | Real-time streaming tool execution |
| REST | `/tools`, `/route`, `/compose`, `/execute`, `/health` | Direct HTTP tool routing & execution |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Protocol Layer                  │
│         A2A  ·  MCP  ·  WebSocket  ·  REST      │
├─────────────────────────────────────────────────┤
│               Router / Composer                  │
│   FunctionGemma selects tools, composes chains   │
├─────────────────────────────────────────────────┤
│                 Tool Registry                    │
│    Dynamic registry with JSON Schema defs        │
│    HTTP · Custom tools                           │
└─────────────────────────────────────────────────┘
```

## Training Pipeline

The full training pipeline generates, downloads, converts, fine-tunes, evaluates, and exports:

```bash
make download    # Download xlam-60k + irrelevance-7.5k
make convert     # Convert combined data to FunctionGemma format
make train       # Fine-tune with LoRA via PEFT/TRL
make benchmark   # Compare base vs fine-tuned model (lm-evaluation-harness)
make export      # Export to GGUF for Ollama
make publish-hf  # Push model to Hugging Face Hub
```

### Training Data

| Source | Examples | Purpose |
|--------|----------|---------|
| Salesforce xlam-function-calling-60k | ~10,000 | General function calling |
| MadeAgents xlam-irrelevance-7.5k | ~3,000 | Negative / refusal examples |
| **Total** | **~13,000** | |

### Fine-tuning Details

- **Base model**: `unsloth/functiongemma-270m-it` (Gemma 3 270M)
- **Method**: LoRA (r=16, alpha=32) via PEFT + TRL SFTTrainer
- **Hardware**: NVIDIA H100 SXM 80GB
- **Format**: FunctionGemma native control tokens

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_AGENT_PORT` | `8888` | Server port |
| `TOOL_AGENT_HOST` | `0.0.0.0` | Bind address |
| `TOOL_AGENT_MODEL` | `functiongemma` | Ollama model name |
| `TOOL_AGENT_BACKEND` | `ollama` | Backend (`ollama` or `transformers`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |

## Project Structure

```
tool_agent/
├── agent/                  # Runtime agent
│   ├── server.py           # FastAPI application
│   ├── model.py            # Inference backends (Ollama, Transformers)
│   ├── router.py           # Tool selection logic
│   ├── composer.py         # Multi-step tool composition
│   ├── tool_registry.py    # Dynamic tool registry
│   ├── config.py           # Environment-based config
│   ├── protocols/          # Protocol implementations
│   │   ├── a2a.py          # Agent-to-Agent
│   │   ├── mcp.py          # Model Context Protocol
│   │   └── websocket.py    # WebSocket streaming
│   └── tools/              # Built-in tools
│       ├── base.py         # Tool base classes
│       └── http.py         # HTTP tool executor
├── training/               # Training pipeline
│   ├── finetune.py         # LoRA fine-tuning with TRL
│   ├── benchmark.py        # lm-evaluation-harness benchmarking
│   ├── evaluate.py         # Standalone evaluation
│   ├── publish_hf.py       # Hugging Face Hub publishing
│   ├── prepare_test_data.py
│   ├── configs/            # Training hyperparameters
│   ├── tasks/              # lm-eval task definitions
│   └── data/               # Training data (gitignored)
├── docker/                 # Container deployment
├── tests/                  # Test suite
├── pyproject.toml          # Python package config
└── Makefile                # CLI interface
```

## License

Apache 2.0
