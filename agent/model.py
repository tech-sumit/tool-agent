"""Model inference backends for tool calling.

Supports four backends:
  - Ollama: for production deployment with GGUF models (xLAM-2, Hammer, etc.)
  - Transformers: for direct HuggingFace model loading (dev/testing)
  - Gemini: Google Gemini API with native function calling
  - Mock: deterministic keyword-matching for tests/demos

xLAM-2 models output standard JSON tool call arrays:
  [{"name": "tool_name", "arguments": {"arg1": "value1"}}]
"""

from __future__ import annotations

import json
import logging
import os
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
    """HuggingFace transformers backend for direct model loading.

    Supports both plain models and PEFT/LoRA adapters. If model_path
    contains an adapter_config.json, loads the base model and merges
    the adapter automatically.

    Detects FunctionGemma models (gemma-based with legacy control tokens)
    and uses the correct system prompt format.
    """

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | None = None,
    ):
        self.model_path = model_path or config.model_name
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        self._is_functiongemma = False

    def _load(self):
        if self._model is not None:
            return

        from pathlib import Path

        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_path = self.model_path
        adapter = self.adapter_path

        # Auto-detect LoRA adapter: if model_path has adapter_config.json,
        # read base_model from it and treat model_path as the adapter
        model_dir = Path(load_path)
        adapter_cfg = model_dir / "adapter_config.json" if model_dir.is_dir() else None
        if adapter is None and adapter_cfg and adapter_cfg.exists():
            cfg = json.loads(adapter_cfg.read_text())
            base = cfg.get("base_model_name_or_path", "")
            if base:
                logger.info("Detected LoRA adapter at %s (base: %s)", load_path, base)
                adapter = str(model_dir)
                load_path = base

        logger.info("Loading model from %s ...", load_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            adapter or load_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            load_path, device_map="auto", trust_remote_code=True
        )

        if adapter:
            from peft import PeftModel

            logger.info("Merging LoRA adapter from %s ...", adapter)
            self._model = PeftModel.from_pretrained(self._model, adapter)
            self._model = self._model.merge_and_unload()
            logger.info("Adapter merged.")

        self._is_functiongemma = "functiongemma" in load_path.lower() or "gemma" in load_path.lower()
        if self._is_functiongemma:
            logger.info("FunctionGemma model detected — using legacy prompt format")

        logger.info("Model loaded.")

    def _build_prompt(self, user_message: str, tools: list[dict], system_prompt: str | None) -> str:
        """Build a raw prompt string for FunctionGemma (Gemma 3 turn markers)."""
        tools_text = _format_tools_for_system(tools)
        developer = f"{LEGACY_SYSTEM_PROMPT}\n\n{tools_text}" if tools else LEGACY_SYSTEM_PROMPT

        return (
            f"<start_of_turn>user\n"
            f"{developer}\n\n"
            f"{user_message}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    async def generate(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        self._load()

        if self._is_functiongemma:
            prompt = self._build_prompt(user_message, tools, system_prompt)
            inputs = self._tokenizer(prompt, return_tensors="pt")
        else:
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
            do_sample=temperature > 0,
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


class GeminiBackend(InferenceBackend):
    """Google Gemini REST API backend with native function calling.

    Uses the Generative Language API directly via httpx — no extra SDK needed.
    Recommended model for lightweight use: gemini-2.0-flash-lite.
    """

    API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._client = httpx.AsyncClient(timeout=120.0)

    def _convert_type(self, type_str: str) -> str:
        return type_str.upper() if type_str else "STRING"

    def _convert_schema(self, schema: dict) -> dict:
        """Convert JSON Schema to Gemini Schema format (uppercase types)."""
        schema_type = schema.get("type", "string")
        result: dict[str, Any] = {"type": self._convert_type(schema_type)}
        if "properties" in schema:
            result["properties"] = {
                k: self._convert_schema(v) for k, v in schema["properties"].items()
            }
        if "required" in schema:
            result["required"] = schema["required"]
        if "description" in schema:
            result["description"] = schema["description"]
        if "enum" in schema:
            result["enum"] = schema["enum"]
        if "items" in schema:
            result["items"] = self._convert_schema(schema["items"])
        elif schema_type == "array":
            result["items"] = {"type": "STRING"}
        return result

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert our function schemas to Gemini's functionDeclarations."""
        declarations = []
        for tool in tools:
            func = tool.get("function", tool)
            params = func.get("parameters", {})
            decl: dict[str, Any] = {
                "name": func["name"],
                "description": func.get("description", ""),
            }
            if params.get("properties"):
                decl["parameters"] = self._convert_schema(params)
            declarations.append(decl)
        return [{"functionDeclarations": declarations}]

    def _parse_response(self, data: dict) -> str:
        candidates = data.get("candidates", [])
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        function_calls = []
        text_parts = []

        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                function_calls.append({
                    "name": fc["name"],
                    "arguments": dict(fc.get("args", {})),
                })
            elif "text" in part:
                text_parts.append(part["text"])

        if function_calls:
            return json.dumps(function_calls)
        return "\n".join(text_parts)

    async def generate(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            body["tools"] = self._convert_tools(tools)
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        url = f"{self.API_BASE}/models/{self.model_name}:generateContent"
        resp = await self._client.post(url, json=body, params={"key": self.api_key})
        resp.raise_for_status()
        return self._parse_response(resp.json())

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Multi-turn generation for the ToolComposer."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            body["tools"] = self._convert_tools(tools)
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        url = f"{self.API_BASE}/models/{self.model_name}:generateContent"
        resp = await self._client.post(url, json=body, params={"key": self.api_key})
        resp.raise_for_status()
        return self._parse_response(resp.json())

    async def health_check(self) -> bool:
        try:
            url = f"{self.API_BASE}/models/{self.model_name}"
            resp = await self._client.get(url, params={"key": self.api_key})
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()


class MockBackend(InferenceBackend):
    """Deterministic backend for testing and demos without a live model.

    Matches tool names from the system prompt against keywords in the
    user message and returns a well-formed JSON tool call array.
    When no tool matches, returns a refusal string.
    """

    async def generate(
        self,
        user_message: str,
        tools: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        msg_lower = user_message.lower()
        matched = []
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "")
            params = func.get("parameters", {}).get("properties", {})
            args = {}
            for pname, pdef in params.items():
                ptype = pdef.get("type", "string")
                if ptype == "string":
                    args[pname] = f"<mock_{pname}>"
                elif ptype in ("integer", "number"):
                    args[pname] = 42
                elif ptype == "boolean":
                    args[pname] = True
                elif ptype == "object":
                    args[pname] = {}
                else:
                    args[pname] = f"<mock_{pname}>"

            keywords = name.replace("_", " ").split()
            if any(kw in msg_lower for kw in keywords):
                matched.append({"name": name, "arguments": args})

        if not matched and tools:
            func = tools[0].get("function", tools[0])
            name = func.get("name", "unknown")
            matched.append({"name": name, "arguments": {}})

        if matched:
            return json.dumps(matched)
        return "I don't have a suitable tool for that request."

    async def health_check(self) -> bool:
        return True


def create_backend(backend_type: str | None = None, **kwargs) -> InferenceBackend:
    """Factory to create the configured inference backend."""
    backend = backend_type or config.model_backend

    if backend == "ollama":
        return OllamaBackend(**kwargs)
    elif backend == "transformers":
        return TransformersBackend(**kwargs)
    elif backend == "gemini":
        return GeminiBackend(**kwargs)
    elif backend == "mock":
        return MockBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Use 'ollama', 'transformers', 'gemini', or 'mock'."
        )
