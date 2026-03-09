"""Evaluate function-calling model accuracy.

Tests models via Ollama against a held-out test set, measuring:
  - Tool name accuracy (did it pick the right tool?)
  - Negative rejection (did it correctly refuse non-tool queries?)

Supports both FunctionGemma and xLAM-2 output formats.

Usage:
    python -m training.evaluate \
        --dataset training/data/training_data.jsonl \
        --base-model functiongemma
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

FUNCTION_CALL_PATTERN = re.compile(
    r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>",
    re.DOTALL,
)
PARAM_PATTERN = re.compile(r"(\w+):<escape>(.*?)<escape>")

NO_TOOL_PHRASES = [
    "none of the available tools",
    "don't have a tool",
    "doesn't match any",
    "no suitable tool",
    "not suitable for this",
    "cannot help with that",
    "i can't",
    "i cannot",
]


def load_dataset(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def parse_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from model output — supports both JSON and FunctionGemma formats."""
    text = text.strip()

    # FunctionGemma control-token format
    fg_matches = list(FUNCTION_CALL_PATTERN.finditer(text))
    if fg_matches:
        calls = []
        for m in fg_matches:
            name = m.group(1)
            params_str = m.group(2)
            arguments = {}
            for pm in PARAM_PATTERN.finditer(params_str):
                arguments[pm.group(1)] = pm.group(2)
            calls.append({"name": name, "arguments": arguments})
        return calls

    # JSON array format (xLAM-2 style)
    try:
        if text.startswith("["):
            calls = json.loads(text)
            if isinstance(calls, list):
                return [c for c in calls if isinstance(c, dict) and "name" in c]
    except json.JSONDecodeError:
        pass

    # Embedded JSON in text
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            calls = json.loads(json_match.group(0))
            if isinstance(calls, list):
                return [c for c in calls if isinstance(c, dict) and "name" in c]
        except json.JSONDecodeError:
            pass

    return []


def is_no_tool_response(text: str) -> bool:
    """Check if the response explicitly declines to use tools."""
    lower = text.lower().strip()
    return any(phrase in lower for phrase in NO_TOOL_PHRASES)


def extract_expected(example: dict) -> tuple[list[str], bool]:
    """Get expected tool names and whether this is a negative example."""
    category = example.get("category", "")

    if category in ("negative", "irrelevance") or "nocall" in category:
        return [], True

    # FunctionGemma pre-formatted text field
    if "text" in example:
        text = example["text"]
        fg_matches = list(FUNCTION_CALL_PATTERN.finditer(text))
        if fg_matches:
            return [m.group(1) for m in fg_matches], False
        if is_no_tool_response(text):
            return [], True
        return [], category == "discovery"

    # xLAM-2 messages format
    messages = example.get("messages", [])
    assistant_msg = None
    for msg in messages:
        if msg.get("role") == "assistant":
            assistant_msg = msg["content"]

    if not assistant_msg:
        return [], False

    calls = parse_tool_calls(assistant_msg)
    if calls:
        return [c["name"] for c in calls], False

    if is_no_tool_response(assistant_msg):
        return [], True

    return [], category == "discovery"


def extract_user_query(example: dict) -> str:
    """Extract user query from either format."""
    if "text" in example:
        text = example["text"]
        user_start = text.find("<start_of_turn>user\n")
        if user_start >= 0:
            content_start = user_start + len("<start_of_turn>user\n")
            content_end = text.find("<end_of_turn>", content_start)
            if content_end >= 0:
                return text[content_start:content_end].strip()

    messages = example.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def extract_system_or_tools(example: dict) -> list[dict]:
    """Extract the messages to send to the model (system + user, no assistant)."""
    if "text" in example:
        text = example["text"]
        user_start = text.find("<start_of_turn>user\n")
        if user_start >= 0:
            content_start = user_start + len("<start_of_turn>user\n")
            content_end = text.find("<end_of_turn>", content_start)
            if content_end >= 0:
                full_user = text[content_start:content_end].strip()
                return [{"role": "user", "content": full_user}]

    messages = example.get("messages", [])
    return [m for m in messages if m["role"] != "assistant"]


def evaluate_with_ollama(
    examples: list[dict],
    model_name: str,
    base_url: str = "http://localhost:11434",
    max_examples: int = 100,
) -> dict:
    """Evaluate model via Ollama API."""
    results = Counter()
    category_results: dict[str, Counter] = defaultdict(Counter)
    errors: list[dict] = []
    latencies: list[float] = []
    predictions: list[dict] = []

    client = httpx.Client(timeout=120.0)

    for i, ex in enumerate(examples[:max_examples]):
        category = ex.get("category", "unknown")
        expected_names, is_negative = extract_expected(ex)
        messages = extract_system_or_tools(ex)

        t0 = time.time()
        try:
            resp = client.post(f"{base_url}/api/chat", json={
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            })
            resp.raise_for_status()
            output = resp.json().get("message", {}).get("content", "")
        except Exception as exc:
            results["error"] += 1
            category_results[category]["error"] += 1
            errors.append({"idx": i, "error": str(exc)})
            continue
        latency = time.time() - t0
        latencies.append(latency)

        predicted_calls = parse_tool_calls(output)
        predicted_names = [c["name"] for c in predicted_calls]
        no_tool = is_no_tool_response(output) and not predicted_names

        results["total"] += 1
        category_results[category]["total"] += 1

        correct = False
        match_type = "miss"

        if is_negative:
            if not predicted_names:
                correct = True
                match_type = "correct"
            else:
                match_type = "miss"
        elif expected_names:
            if predicted_names == expected_names:
                correct = True
                match_type = "correct"
            elif set(predicted_names) == set(expected_names):
                match_type = "set_match"
            elif predicted_names and predicted_names[0] == expected_names[0]:
                match_type = "first_match"
            else:
                match_type = "miss"
        else:
            if not predicted_names:
                correct = True
                match_type = "correct"
            else:
                match_type = "miss"

        results[match_type] += 1
        category_results[category][match_type] += 1

        predictions.append({
            "idx": i,
            "category": category,
            "expected": expected_names if expected_names else "no_tool",
            "predicted": predicted_names if predicted_names else "no_tool",
            "correct": correct,
            "output_preview": output[:200],
        })

        if match_type == "miss" and len(errors) < 20:
            errors.append({
                "idx": i, "category": category,
                "expected": expected_names if expected_names else "no_tool",
                "predicted": predicted_names if predicted_names else "no_tool",
                "output": output[:200],
            })

        if (i + 1) % 10 == 0:
            total = results["total"]
            acc = results["correct"] / total * 100 if total else 0
            avg_lat = sum(latencies[-10:]) / min(10, len(latencies))
            print(f"  [{i + 1}/{min(len(examples), max_examples)}] "
                  f"Accuracy: {acc:.1f}% | Avg latency: {avg_lat:.2f}s")

    client.close()
    return {
        "results": dict(results),
        "by_category": {k: dict(v) for k, v in category_results.items()},
        "errors": errors,
        "predictions": predictions,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
    }


def evaluate_with_transformers(
    examples: list[dict],
    model_path: str,
    adapter_path: str | None = None,
    max_examples: int = 100,
) -> dict:
    """Evaluate model via HuggingFace Transformers (local inference)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"  Loading model: {model_path} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"  Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    results = Counter()
    category_results: dict[str, Counter] = defaultdict(Counter)
    errors: list[dict] = []
    latencies: list[float] = []
    predictions: list[dict] = []

    for i, ex in enumerate(examples[:max_examples]):
        category = ex.get("category", "unknown")
        expected_names, is_negative = extract_expected(ex)

        if "text" in ex:
            text = ex["text"]
            model_start = text.find("<start_of_turn>model\n")
            if model_start >= 0:
                prompt = text[:model_start + len("<start_of_turn>model\n")]
            else:
                prompt = text
        else:
            messages = extract_system_or_tools(ex)
            prompt = "\n".join(f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>" for m in messages)
            prompt += "\n<start_of_turn>model\n"

        t0 = time.time()
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=896).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=256, temperature=0.1, top_p=0.95,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
            output = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        except Exception as exc:
            results["error"] += 1
            category_results[category]["error"] += 1
            errors.append({"idx": i, "error": str(exc)})
            continue
        latency = time.time() - t0
        latencies.append(latency)

        predicted_calls = parse_tool_calls(output)
        predicted_names = [c["name"] for c in predicted_calls]
        no_tool = is_no_tool_response(output) and not predicted_names

        results["total"] += 1
        category_results[category]["total"] += 1

        correct = False
        match_type = "miss"

        if is_negative:
            if not predicted_names:
                correct = True
                match_type = "correct"
        elif expected_names:
            if predicted_names == expected_names:
                correct = True
                match_type = "correct"
            elif set(predicted_names) == set(expected_names):
                match_type = "set_match"
            elif predicted_names and predicted_names[0] == expected_names[0]:
                match_type = "first_match"
        else:
            if not predicted_names:
                correct = True
                match_type = "correct"

        results[match_type] += 1
        category_results[category][match_type] += 1

        predictions.append({
            "idx": i,
            "category": category,
            "expected": expected_names if expected_names else "no_tool",
            "predicted": predicted_names if predicted_names else "no_tool",
            "correct": correct,
            "output_preview": output[:200],
        })

        if match_type == "miss" and len(errors) < 20:
            errors.append({
                "idx": i, "category": category,
                "expected": expected_names if expected_names else "no_tool",
                "predicted": predicted_names if predicted_names else "no_tool",
                "output": output[:200],
            })

        if (i + 1) % 10 == 0:
            total = results["total"]
            acc = results["correct"] / total * 100 if total else 0
            avg_lat = sum(latencies[-10:]) / min(10, len(latencies))
            print(f"  [{i + 1}/{min(len(examples), max_examples)}] "
                  f"Accuracy: {acc:.1f}% | Avg latency: {avg_lat:.2f}s")

    return {
        "results": dict(results),
        "by_category": {k: dict(v) for k, v in category_results.items()},
        "errors": errors,
        "predictions": predictions,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
    }


def print_results(label: str, data: dict):
    results = data["results"]
    total = results.get("total", 0)
    if total == 0:
        print(f"\n{label}: No examples evaluated")
        return

    correct = results.get("correct", 0)
    set_match = results.get("set_match", 0)
    first_match = results.get("first_match", 0)
    miss = results.get("miss", 0)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total examples:  {total}")
    print(f"  Exact match:     {correct}/{total} ({correct / total * 100:.1f}%)")
    print(f"  Set match:       {set_match}/{total} ({set_match / total * 100:.1f}%)")
    print(f"  First match:     {first_match}/{total} ({first_match / total * 100:.1f}%)")
    print(f"  Miss:            {miss}/{total} ({miss / total * 100:.1f}%)")
    print(f"  Avg latency:     {data['avg_latency']:.2f}s")

    print(f"\n  By Category:")
    for cat, cat_res in sorted(data["by_category"].items()):
        cat_total = cat_res.get("total", 0)
        cat_correct = cat_res.get("correct", 0)
        cat_acc = cat_correct / cat_total * 100 if cat_total else 0
        print(f"    {cat:30s}: {cat_correct}/{cat_total} ({cat_acc:.1f}%)")

    if data["errors"]:
        print(f"\n  Sample errors ({len(data['errors'])} total):")
        for err in data["errors"][:5]:
            if "error" in err and "expected" not in err:
                print(f"    [{err['idx']}] ERROR: {err['error'][:100]}")
            else:
                print(f"    [{err['idx']}] {err.get('category', '?')}: "
                      f"expected={err.get('expected')}, "
                      f"predicted={err.get('predicted')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate function-calling accuracy")
    parser.add_argument("--dataset", required=True, type=Path, help="Test data JSONL")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model name for fine-tuned model")
    parser.add_argument("--base-model", type=str, default="functiongemma",
                        help="Base model in Ollama (default: functiongemma)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    print(f"Loading dataset from {args.dataset}...")
    all_examples = load_dataset(args.dataset)
    random.shuffle(all_examples)

    cats = defaultdict(int)
    for ex in all_examples:
        cats[ex.get("category", "?")] += 1
    print(f"  {len(all_examples)} total examples")
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    test_examples = all_examples[:args.max_examples]
    print(f"\nEvaluating on {len(test_examples)} examples...")

    print(f"\n--- Evaluating BASE model: {args.base_model} ---")
    base_results = evaluate_with_ollama(
        test_examples, args.base_model, args.ollama_url, args.max_examples,
    )
    print_results(f"BASE: {args.base_model}", base_results)

    if args.model:
        print(f"\n--- Evaluating FINE-TUNED model: {args.model} ---")
        ft_results = evaluate_with_ollama(
            test_examples, args.model, args.ollama_url, args.max_examples,
        )
        print_results(f"FINE-TUNED: {args.model}", ft_results)

        base_acc = (base_results["results"].get("correct", 0) /
                    max(base_results["results"].get("total", 1), 1) * 100)
        ft_acc = (ft_results["results"].get("correct", 0) /
                  max(ft_results["results"].get("total", 1), 1) * 100)
        print(f"\n{'=' * 60}")
        print(f"  COMPARISON")
        print(f"{'=' * 60}")
        print(f"  Base accuracy:       {base_acc:.1f}%")
        print(f"  Fine-tuned accuracy: {ft_acc:.1f}%")
        print(f"  Improvement:         {ft_acc - base_acc:+.1f}%")


if __name__ == "__main__":
    main()
