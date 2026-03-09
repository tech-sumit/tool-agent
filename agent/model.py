"""Model inference backends for tool calling.

Supports two backends:
  - Ollama: for production deployment with GGUF models (xLAM-2, Hammer, etc.)
  - Transformers: for direct HuggingFace model loading (dev/testing)

xLAM-2 models output standard JSON tool call arrays:
  [{"name": "tool_name", "arguments": {"arg1": "value1"}}]
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from agent.config import config

logger = logging.getLogger(__name__)

FUNCTION_CALL_PATTERN = re.compile(
    r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>",
    re.DOTALL,
)

PARAM_PATTERN = re.compile(r"(\w+):<escape>(.*?)<escape>")

XLAM_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools.\n"
    "You have access to a set of tools. When using tools, make calls "
    "in a single JSON array:\n\n"
    '[{"name": "tool_call_name", "arguments": {"arg1": "value1", '
    '"arg2": "value2"}}, ... (additional parallel tool calls as needed)]\n\n'
    "If no tool is suitable, state that explicitly. If the user's input "
    "lacks required parameters, ask for clarification. Do not interpret "
    "or respond until tool results are returned. Once they are available, "
    "process them or make additional calls if needed. For tasks that don't "
    "require tools, such as casual conversation or general advice, respond "
    "directly in plain text. The available tools are:"
)

LEGACY_SYSTEM_PROMPT = (
    "You are a model that can do function calling with the following functions"
)


@dataclass
class FunctionCall:
    """Parsed function call from model output."""

    name: str
    arguments: dict[str, Any]

    @classmethod
    def parse(cls, text: str) -> list[FunctionCall]:
        """Parse function calls from model output.

        Supports both xLAM-2 JSON array format and legacy FunctionGemma format.
        Tries JSON first (including embedded arrays), falls back to legacy.
        """
        text = text.strip()
        if text.startswith("["):
            return cls._parse_json(text)
        if "[{" in text:
            result = cls._parse_json(text)
            if result:
                return result
        return cls._parse_legacy(text)

    @classmethod
    def _parse_json(cls, text: str) -> list[FunctionCall]:
        """Parse xLAM-2 / standard JSON tool call array."""
        try:
            calls_data = json.loads(text)
            if not isinstance(calls_data, list):
                calls_data = [calls_data]
            return [
                cls(
                    name=c.get("name", ""),
                    arguments=c.get("arguments", {}),
                )
                for c in calls_data
                if isinstance(c, dict) and c.get("name")
            ]
        except json.JSONDecodeError:
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                try:
                    return cls._parse_json(json_match.group(0))
                except Exception:
                    pass
            return []

    @classmethod
    def _parse_legacy(cls, text: str) -> list[FunctionCall]:
        """Parse legacy FunctionGemma format."""
        calls = []
        for match in FUNCTION_CALL_PATTERN.finditer(text):
            name = match.group(1)
            params_str = match.group(2)
            arguments: dict[str, Any] = {}
            for param_match in PARAM_PATTERN.finditer(params_str):
                arguments[param_match.group(1)] = param_match.group(2)
            calls.append(cls(name=name, arguments=arguments))
        return calls

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


class InferenceBackend(ABC):
    """Abstract inference backend."""

    @abstractmethod
    async def generate(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate raw text output from the model."""
        ...

    async def call_functions(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
    ) -> list[FunctionCall]:
        """Generate and parse function calls."""
        raw = await self.generate(user_message, tools, system_prompt, max_tokens)
        return FunctionCall.parse(raw)

    @abstractmethod
    async def health_check(self) -> bool:
        ...


def _format_tools_for_system(tools: list[dict]) -> str:
    """Format tool schemas as individual JSON objects for the system prompt."""
    parts = []
    for tool in tools:
        func = tool.get("function", tool)
        parts.append(json.dumps(func, indent=2))
    return "\n\n".join(parts)


class OllamaBackend(InferenceBackend):
    """Ollama REST API backend for GGUF model inference."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        self.model_name = model_name or config.model_name
        self._client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        system = system_prompt or XLAM_SYSTEM_PROMPT
        tools_text = _format_tools_for_system(tools)
        full_system = f"{system}\n\n{tools_text}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate with full conversation history for multi-turn tool use."""
        system = system_prompt or XLAM_SYSTEM_PROMPT
        tools_text = _format_tools_for_system(tools)
        full_system = f"{system}\n\n{tools_text}"

        all_messages = [{"role": "system", "content": full_system}] + messages

        payload = {
            "model": self.model_name,
            "messages": all_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()


class TransformersBackend(InferenceBackend):
    """HuggingFace transformers backend for direct model loading."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or config.model_name
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model from %s ...", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="auto"
        )
        logger.info("Model loaded.")

    async def generate(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        self._load()

        system = system_prompt or XLAM_SYSTEM_PROMPT
        tools_text = _format_tools_for_system(tools)
        full_system = f"{system}\n\n{tools_text}"

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_message},
        ]

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        outputs = self._model.generate(
            **inputs.to(self._model.device),
            max_new_tokens=max_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
            temperature=temperature,
        )

        return self._tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )

    async def health_check(self) -> bool:
        try:
            self._load()
            return self._model is not None
        except Exception:
            return False


def create_backend(backend_type: str | None = None, **kwargs) -> InferenceBackend:
    """Factory to create the configured inference backend."""
    backend = backend_type or config.model_backend

    if backend == "ollama":
        return OllamaBackend(**kwargs)
    elif backend == "transformers":
        return TransformersBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'transformers'.")
