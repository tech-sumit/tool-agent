"""Agent configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    host: str = field(default_factory=lambda: os.getenv("TOOL_AGENT_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("TOOL_AGENT_PORT", "8888")))

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "TOOL_AGENT_MODEL",
            "functiongemma",
        )
    )
    model_backend: str = field(
        default_factory=lambda: os.getenv("TOOL_AGENT_BACKEND", "ollama")
    )

    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    n8n_api_url: str = field(
        default_factory=lambda: os.getenv("N8N_API_URL", "http://localhost:5678")
    )
    n8n_api_key: str = field(default_factory=lambda: os.getenv("N8N_API_KEY", ""))

    agent_name: str = field(
        default_factory=lambda: os.getenv("TOOL_AGENT_NAME", "Tool Agent")
    )
    agent_description: str = field(
        default_factory=lambda: os.getenv(
            "TOOL_AGENT_DESCRIPTION",
            "FunctionGemma powered integration expert for tool routing, "
            "composition, and discovery across n8n integrations and APIs.",
        )
    )
    agent_version: str = "0.1.0"


config = Config()
