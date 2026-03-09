"""Download and prepare external function-calling datasets.

Downloads from HuggingFace:
  1. Salesforce/xlam-function-calling-60k — 60K xLAM-native examples (gated)
  2. MadeAgents/xlam-irrelevance-7.5k — 7.5K negative/no-call examples
  Fallback: younissk/tool-calling-mix (ungated, contains xlam60k subset)

Converts everything to the xLAM-2 ChatML format and merges with
any existing n8n-specific training data.

Usage:
    HF_TOKEN=hf_xxx python -m training.download_datasets
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


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

NO_TOOL_RESPONSES = [
    "None of the available tools are suitable for this request.",
    "I don't have a tool that can help with this specific request.",
    "This request doesn't match any of my available tools.",
    "No suitable tool is available for this query. Let me help directly.",
]


def format_tools_text(tools_raw: str | list) -> str:
    """Format tool definitions for the system message."""
    if isinstance(tools_raw, str):
        try:
            tools = json.loads(tools_raw)
        except json.JSONDecodeError:
            return tools_raw
    else:
        tools = tools_raw

    if not isinstance(tools, list):
        return str(tools)

    return "\n\n".join(json.dumps(t) for t in tools)


def convert_xlam_60k(row: dict) -> dict | None:
    """Convert a Salesforce/xlam-function-calling-60k row to ChatML messages."""
    query = row.get("query", "")
    tools_raw = row.get("tools", "[]")
    answers_raw = row.get("answers", "[]")

    if not query:
        return None

    tools_text = format_tools_text(tools_raw)

    if isinstance(answers_raw, str):
        try:
            answers = json.loads(answers_raw)
        except json.JSONDecodeError:
            answers = []
    else:
        answers = answers_raw

    if not isinstance(answers, list):
        answers = [answers] if answers else []

    assistant_content = json.dumps(answers) if answers else random.choice(NO_TOOL_RESPONSES)

    return {
        "messages": [
            {"role": "system", "content": f"{XLAM_SYSTEM_PROMPT}\n\n{tools_text}"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_content},
        ],
        "category": "xlam_60k",
    }


def convert_tool_calling_mix(row: dict) -> dict | None:
    """Convert a younissk/tool-calling-mix row to ChatML messages."""
    msgs_raw = row.get("messages_json", "[]")
    tools_raw = row.get("tools_json", "[]")
    target_raw = row.get("target_json", "{}")
    source = row.get("meta_source", "unknown")
    difficulty = row.get("difficulty", "simple")

    try:
        msgs = json.loads(msgs_raw) if isinstance(msgs_raw, str) else msgs_raw
        target = json.loads(target_raw) if isinstance(target_raw, str) else target_raw
    except json.JSONDecodeError:
        return None

    tools_text = format_tools_text(tools_raw)

    user_msg = None
    for m in (msgs if isinstance(msgs, list) else []):
        if m.get("role") == "user" and m.get("content"):
            user_msg = m["content"].strip()
            break

    if not user_msg:
        return None

    tool_calls = target.get("tool_calls", []) if isinstance(target, dict) else []

    if difficulty == "no_call" or not tool_calls:
        assistant_content = random.choice(NO_TOOL_RESPONSES)
    else:
        clean_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict) and "name" in tc:
                clean_calls.append({
                    "name": tc["name"],
                    "arguments": tc.get("arguments", {}),
                })
        assistant_content = json.dumps(clean_calls) if clean_calls else random.choice(NO_TOOL_RESPONSES)

    return {
        "messages": [
            {"role": "system", "content": f"{XLAM_SYSTEM_PROMPT}\n\n{tools_text}"},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_content},
        ],
        "category": f"tcm_{source}",
    }


def convert_irrelevance_example(row: dict) -> dict | None:
    """Convert an xlam-irrelevance-7.5k row to ChatML messages."""
    query = row.get("query", "")
    tools_raw = row.get("tools", "[]")

    if not query:
        return None

    tools_text = format_tools_text(tools_raw)

    return {
        "messages": [
            {"role": "system", "content": f"{XLAM_SYSTEM_PROMPT}\n\n{tools_text}"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": random.choice(NO_TOOL_RESPONSES)},
        ],
        "category": "irrelevance",
    }


def main():
    parser = argparse.ArgumentParser(description="Download and prepare training datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("training/data"))
    parser.add_argument("--n8n-data", type=Path, default=None,
                        help="Path to n8n-specific training data JSONL to merge")
    parser.add_argument("--xlam-sample", type=int, default=10000,
                        help="Max examples from tool-calling-mix (default: 10000)")
    parser.add_argument("--irrelevance-sample", type=int, default=3000,
                        help="Max examples from irrelevance dataset (default: 3000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_examples: list[dict] = []
    hf_token = os.environ.get("HF_TOKEN")

    # 1. Download Salesforce/xlam-function-calling-60k (gated, needs token)
    xlam_ok = False
    if hf_token:
        print("Downloading Salesforce/xlam-function-calling-60k...")
        try:
            ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train",
                              token=hf_token)
            print(f"  Downloaded {len(ds)} examples")

            converted = []
            for row in ds:
                ex = convert_xlam_60k(row)
                if ex:
                    converted.append(ex)

            if len(converted) > args.xlam_sample:
                random.shuffle(converted)
                converted = converted[:args.xlam_sample]

            print(f"  Converted {len(converted)} examples (sampled to {args.xlam_sample})")
            all_examples.extend(converted)

            xlam_path = args.output_dir / "xlam_60k.jsonl"
            with open(xlam_path, "w") as f:
                for ex in converted:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  Saved to {xlam_path}")
            xlam_ok = True
        except Exception as e:
            print(f"  ERROR downloading xlam-60k: {e}")
    else:
        print("No HF_TOKEN set, skipping gated Salesforce/xlam-function-calling-60k")

    # 1b. Fallback: younissk/tool-calling-mix (ungated, contains xlam60k subset)
    if not xlam_ok:
        print("\nFallback: Downloading younissk/tool-calling-mix (ungated)...")
        try:
            ds = load_dataset("younissk/tool-calling-mix", split="train")
            print(f"  Downloaded {len(ds)} examples")

            sources = Counter(ds["meta_source"])
            print(f"  Sources: {dict(sources.most_common(5))}")

            converted = []
            for row in ds:
                ex = convert_tool_calling_mix(row)
                if ex:
                    converted.append(ex)

            if len(converted) > args.xlam_sample:
                random.shuffle(converted)
                converted = converted[:args.xlam_sample]

            print(f"  Converted {len(converted)} examples (sampled to {args.xlam_sample})")
            all_examples.extend(converted)

            tcm_path = args.output_dir / "tool_calling_mix.jsonl"
            with open(tcm_path, "w") as f:
                for ex in converted:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  Saved to {tcm_path}")
        except Exception as e:
            print(f"  ERROR downloading tool-calling-mix: {e}")

    # 2. Download xlam-irrelevance-7.5k
    print("\nDownloading MadeAgents/xlam-irrelevance-7.5k...")
    try:
        ds_irr = load_dataset("MadeAgents/xlam-irrelevance-7.5k", split="train")
        print(f"  Downloaded {len(ds_irr)} examples")

        irr_examples = []
        for row in ds_irr:
            ex = convert_irrelevance_example(row)
            if ex:
                irr_examples.append(ex)

        if len(irr_examples) > args.irrelevance_sample:
            random.shuffle(irr_examples)
            irr_examples = irr_examples[:args.irrelevance_sample]

        print(f"  Converted {len(irr_examples)} examples (sampled to {args.irrelevance_sample})")
        all_examples.extend(irr_examples)

        irr_path = args.output_dir / "xlam_irrelevance.jsonl"
        with open(irr_path, "w") as f:
            for ex in irr_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Saved to {irr_path}")

    except Exception as e:
        print(f"  ERROR downloading irrelevance dataset: {e}")

    # 3. Merge with n8n-specific data
    n8n_path = args.n8n_data or args.output_dir / "training_data.jsonl"
    if n8n_path.exists():
        print(f"\nLoading n8n-specific data from {n8n_path}...")
        n8n_examples = []
        with open(n8n_path) as f:
            for line in f:
                if line.strip():
                    n8n_examples.append(json.loads(line))
        print(f"  Loaded {len(n8n_examples)} n8n examples")
        all_examples.extend(n8n_examples)
    else:
        print(f"\n  No n8n data at {n8n_path}, skipping merge")

    # 4. Shuffle and write combined dataset
    random.shuffle(all_examples)

    combined_path = args.output_dir / "combined_training_data.jsonl"
    with open(combined_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    cats = Counter(ex.get("category", "unknown") for ex in all_examples)

    print(f"\n{'=' * 60}")
    print(f"Combined dataset: {len(all_examples)} examples")
    print(f"Written to: {combined_path}")
    print(f"{'=' * 60}")
    for cat, count in sorted(cats.items()):
        pct = 100 * count / len(all_examples)
        print(f"  {cat:40s} {count:6d}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
