"""Custom evaluation functions for lm-evaluation-harness.

Parses FunctionGemma control-token output format and computes
function-calling accuracy metrics.
"""

import re

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


def parse_tool_names(text: str) -> list[str]:
    return [m.group(1) for m in FUNCTION_CALL_PATTERN.finditer(text)]


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in FUNCTION_CALL_PATTERN.finditer(text):
        args = {}
        for pm in PARAM_PATTERN.finditer(m.group(2)):
            args[pm.group(1)] = pm.group(2)
        calls.append({"name": m.group(1), "arguments": args})
    return calls


def is_no_tool_response(text: str) -> bool:
    lower = text.lower().strip()
    return any(phrase in lower for phrase in NO_TOOL_PHRASES)


def doc_to_text(doc: dict) -> str:
    """Extract prompt (everything up to and including <start_of_turn>model marker)."""
    text = doc["text"]
    marker = "<start_of_turn>model\n"
    idx = text.find(marker)
    if idx >= 0:
        return text[:idx + len(marker)]
    return text


def doc_to_target(doc: dict) -> str:
    """Extract expected model response."""
    text = doc["text"]
    marker = "<start_of_turn>model\n"
    idx = text.find(marker)
    if idx >= 0:
        target = text[idx + len(marker):]
        end = target.find("<end_of_turn>")
        if end >= 0:
            return target[:end]
        return target
    return ""


def process_results(doc: dict, results: list[str]) -> dict:
    """Score a single prediction against the expected output.

    Returns a dict of metric_name -> float value. lm-eval-harness
    collects these across all docs and aggregates with `mean`.
    """
    generated = results[0].strip()
    target = doc_to_target(doc)
    category = doc.get("category", "unknown")

    expected_names = parse_tool_names(target)
    predicted_names = parse_tool_names(generated)

    is_negative = category in ("negative", "irrelevance", "nocall")

    if is_negative:
        exact = 1.0 if not predicted_names else 0.0
        first = exact
        neg_rej = exact
        param_match = exact
    elif expected_names:
        exact = 1.0 if predicted_names == expected_names else 0.0
        first = 1.0 if (predicted_names and predicted_names[0] == expected_names[0]) else 0.0
        neg_rej = 1.0

        expected_calls = parse_tool_calls(target)
        predicted_calls = parse_tool_calls(generated)
        if expected_calls and predicted_calls and expected_calls[0]["name"] == predicted_calls[0]["name"]:
            exp_args = expected_calls[0]["arguments"]
            pred_args = predicted_calls[0]["arguments"]
            if exp_args:
                matching = sum(1 for k in exp_args if pred_args.get(k) == exp_args[k])
                param_match = matching / len(exp_args)
            else:
                param_match = 1.0
        else:
            param_match = 0.0
    else:
        exact = 1.0 if not predicted_names else 0.0
        first = exact
        neg_rej = 1.0
        param_match = exact

    return {
        "tool_selection_acc": exact,
        "first_tool_acc": first,
        "negative_rejection": neg_rej,
        "param_accuracy": param_match,
    }
