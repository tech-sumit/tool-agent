"""Convert combined training data from xLAM-2 JSON format to FunctionGemma format.

FunctionGemma uses Gemma 3 turn markers with control tokens for function calls:
  <start_of_turn>user
  You are a model that can do function calling...
  [tool schemas]
  [query]<end_of_turn>
  <start_of_turn>model
  <start_function_call>call:fn{k:<escape>v<escape>}<end_function_call><end_of_turn>

Usage:
    python -m training.convert_to_functiongemma \
        --input training/data/combined_training_data.jsonl \
        --output training/data/functiongemma_training.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

XLAM_PROMPT_PREFIX = "You are a helpful assistant that can use tools."

FG_DEVELOPER_PROMPT = (
    "You are a model that can do function calling with the following functions"
)

NO_TOOL_PHRASES = [
    "none of the available tools",
    "don't have a tool",
    "doesn't match any",
    "no suitable tool",
    "no tool is suitable",
    "not suitable for this",
]


def extract_tools_from_system(system_content: str) -> list[dict]:
    """Extract tool schema JSON objects from the xLAM system message."""
    idx = system_content.find("The available tools are:")
    if idx == -1:
        idx = system_content.find("directly in plain text.")
        if idx == -1:
            return []
        idx += len("directly in plain text.")
    else:
        idx += len("The available tools are:")

    tools_text = system_content[idx:].strip()
    if not tools_text:
        return []

    tools = []
    depth = 0
    start = None

    for i, ch in enumerate(tools_text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(tools_text[start : i + 1])
                    tools.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    return tools


def format_tool_schemas_for_fg(tools: list[dict]) -> str:
    """Format tool schemas in a compact way for the developer message."""
    parts = []
    for tool in tools:
        func = tool.get("function", tool)
        parts.append(json.dumps(func))
    return "\n".join(parts)


def json_calls_to_fg(assistant_content: str) -> str | None:
    """Convert JSON tool call array to FunctionGemma control-token format.

    Returns None if the content is a no-tool response.
    """
    content = assistant_content.strip()

    if not content.startswith("["):
        for phrase in NO_TOOL_PHRASES:
            if phrase in content.lower():
                return None
        return None

    try:
        calls = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(calls, list) or not calls:
        return None

    parts = []
    for call in calls:
        if not isinstance(call, dict) or "name" not in call:
            continue
        name = call["name"]
        args = call.get("arguments", {})

        param_parts = []
        for k, v in args.items():
            val_str = str(v) if not isinstance(v, str) else v
            param_parts.append(f"{k}:<escape>{val_str}<escape>")

        params_str = ",".join(param_parts)
        parts.append(f"<start_function_call>call:{name}{{{params_str}}}<end_function_call>")

    return "\n".join(parts) if parts else None


def convert_example(example: dict) -> dict | None:
    """Convert a single xLAM-2 format example to FunctionGemma training text."""
    messages = example.get("messages", [])
    category = example.get("category", "unknown")

    system_msg = ""
    user_msg = ""
    assistant_msg = ""

    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            system_msg = content
        elif role == "user":
            user_msg = content
        elif role == "assistant":
            assistant_msg = content

    if not user_msg:
        return None

    tools = extract_tools_from_system(system_msg)
    tools_text = format_tool_schemas_for_fg(tools) if tools else ""

    fg_output = json_calls_to_fg(assistant_msg)

    if fg_output is not None:
        model_response = fg_output
    else:
        model_response = assistant_msg if assistant_msg else "I cannot help with that using the available tools."

    developer_section = FG_DEVELOPER_PROMPT
    if tools_text:
        developer_section = f"{FG_DEVELOPER_PROMPT}\n\n{tools_text}"

    text = (
        f"<start_of_turn>user\n"
        f"{developer_section}\n\n"
        f"{user_msg}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{model_response}<end_of_turn>"
    )

    return {"text": text, "category": category}


def main():
    parser = argparse.ArgumentParser(
        description="Convert combined training data to FunctionGemma format"
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("training/data/combined_training_data.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("training/data/functiongemma_training.jsonl"),
    )
    args = parser.parse_args()

    print(f"Converting {args.input} -> {args.output}")

    converted = 0
    skipped = 0
    categories: Counter = Counter()
    has_tools = 0
    no_tools = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue

            example = json.loads(line)
            result = convert_example(example)

            if result is None:
                skipped += 1
                continue

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            converted += 1
            categories[result["category"]] += 1

            if "<start_function_call>" in result["text"]:
                has_tools += 1
            else:
                no_tools += 1

    print(f"\nConverted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Tool calls: {has_tools} | No-tool: {no_tools}")
    print(f"\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
