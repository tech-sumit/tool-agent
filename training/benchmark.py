"""Benchmark base FunctionGemma vs fine-tuned model using lm-evaluation-harness.

Uses EleutherAI's lm-eval framework with custom task definitions for
function-calling accuracy evaluation. Runs both base and fine-tuned
models on a general test set and generates a report.

Usage:
    python -m training.benchmark
    python -m training.benchmark --limit 50  # quick run
    python -m training.benchmark --base-model google/functiongemma-270m-it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import torch
import lm_eval
import lm_eval.tasks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = Path(__file__).resolve().parent / "tasks"


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _prepare_tasks(task_dir: Path) -> str:
    """Copy task YAMLs to a temp dir with absolute data_files paths.

    lm-eval resolves data_files relative to an unpredictable location,
    so we rewrite them to absolute paths at runtime.
    """
    tmp = Path(tempfile.mkdtemp(prefix="lmeval_tasks_"))
    eval_dir = tmp / "functiongemma_eval"
    eval_dir.mkdir(parents=True)

    src_eval = task_dir / "functiongemma_eval"

    shutil.copy2(src_eval / "utils.py", eval_dir / "utils.py")

    for yaml_file in src_eval.glob("*.yaml"):
        content = yaml_file.read_text()
        content = content.replace(
            "training/data/general_test.jsonl",
            str((PROJECT_ROOT / "training/data/general_test.jsonl").resolve()),
        )
        (eval_dir / yaml_file.name).write_text(content)

    return str(tmp)


def run_evaluation(
    model_id: str,
    adapter_path: str | None,
    tasks: list[str],
    limit: int | None,
    batch_size: str = "auto",
    device: str | None = None,
) -> dict:
    """Run lm-eval on a model (optionally with PEFT adapter)."""
    dev = device or _detect_device()
    model_args = f"pretrained={model_id},trust_remote_code=True"
    if dev != "cuda":
        model_args += f",device={dev}"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    tmp_task_dir = _prepare_tasks(TASK_DIR)
    try:
        task_manager = lm_eval.tasks.TaskManager(include_path=tmp_task_dir)
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            batch_size=batch_size,
            limit=limit,
            task_manager=task_manager,
            log_samples=True,
        )
    finally:
        shutil.rmtree(tmp_task_dir, ignore_errors=True)
    return results


def extract_metrics(results: dict, task_name: str) -> dict:
    """Pull metrics from lm-eval results dict."""
    task_results = results.get("results", {}).get(task_name, {})
    return {
        "tool_selection_acc": task_results.get("tool_selection_acc,none", 0),
        "first_tool_acc": task_results.get("first_tool_acc,none", 0),
        "negative_rejection": task_results.get("negative_rejection,none", 0),
        "param_accuracy": task_results.get("param_accuracy,none", 0),
    }


def generate_report(
    base_model: str,
    adapter_path: str | None,
    base_results: dict,
    ft_results: dict | None,
    limit: int | None,
    output_path: Path,
):
    """Generate a markdown benchmark report from lm-eval results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    base_general = extract_metrics(base_results, "functiongemma_general")

    lines = [
        "# FunctionGemma Benchmark Report",
        "",
        f"**Generated:** {now}  ",
        f"**Framework:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)  ",
        f"**Samples per task:** {limit or 'all'}",
        "",
        "## Models",
        "",
        "| Model | Type |",
        "|-------|------|",
        f"| `{base_model}` | Base (pre-trained) |",
    ]
    if adapter_path:
        lines.append(f"| `{base_model}` + LoRA (`{adapter_path}`) | Fine-tuned |")

    lines += [
        "",
        "## Metrics",
        "",
        "| Metric | Description |",
        "|--------|-------------|",
        "| **Tool Selection Acc** | Exact match on predicted vs expected tool name(s) |",
        "| **First Tool Acc** | Whether the first predicted tool matches the first expected tool |",
        "| **Negative Rejection** | Correctly refusing to call a tool when none is appropriate |",
        "| **Param Accuracy** | Fraction of parameters correctly predicted (when tool is correct) |",
        "",
    ]

    if ft_results:
        ft_general = extract_metrics(ft_results, "functiongemma_general")

        lines += [
            "## Results",
            "",
            "### General Function Calling",
            "",
            "| Metric | Base | Fine-tuned | Delta |",
            "|--------|------|-----------|-------|",
        ]
        for metric, label in [
            ("tool_selection_acc", "Tool Selection Acc"),
            ("first_tool_acc", "First Tool Acc"),
            ("negative_rejection", "Negative Rejection"),
            ("param_accuracy", "Param Accuracy"),
        ]:
            b = base_general[metric] * 100
            f = ft_general[metric] * 100
            d = f - b
            lines.append(f"| {label} | {b:.1f}% | {f:.1f}% | {d:+.1f}% |")

        gen_delta = (ft_general["tool_selection_acc"] - base_general["tool_selection_acc"]) * 100

        lines += ["", "## Conclusion", ""]
        if gen_delta > 5:
            lines.append(f"Fine-tuning significantly improved tool selection accuracy ({gen_delta:+.1f}%). The fine-tuned model is recommended for production use.")
        elif gen_delta < -5:
            lines.append(f"Fine-tuning degraded performance ({gen_delta:+.1f}%). The base model may be preferable, or training hyperparameters need adjustment.")
        else:
            lines.append(f"Fine-tuning showed modest changes ({gen_delta:+.1f}%). Consider adjusting training data or hyperparameters for larger gains.")
    else:
        lines += [
            "## Results (Base Model Only)",
            "",
            "| Metric | General |",
            "|--------|---------|",
        ]
        for metric, label in [
            ("tool_selection_acc", "Tool Selection Acc"),
            ("first_tool_acc", "First Tool Acc"),
            ("negative_rejection", "Negative Rejection"),
            ("param_accuracy", "Param Accuracy"),
        ]:
            b_gen = base_general[metric] * 100
            lines.append(f"| {label} | {b_gen:.1f}% |")

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nReport written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FunctionGemma (lm-eval-harness)")
    parser.add_argument("--base-model", default="unsloth/functiongemma-270m-it")
    parser.add_argument("--adapter-path", type=Path, default=Path("./models/finetuned"))
    parser.add_argument("--output", type=Path, default=Path("training/reports/benchmark_report.md"))
    parser.add_argument("--limit", type=int, default=None, help="Max examples per task")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    args = parser.parse_args()

    tasks = ["functiongemma_general"]

    print(f"{'=' * 60}")
    print(f"  DEVICE: {_detect_device()}")
    print(f"  BASE MODEL: {args.base_model}")
    print(f"{'=' * 60}")
    base_results = run_evaluation(
        args.base_model, None, tasks, args.limit, args.batch_size,
    )

    m = extract_metrics(base_results, "functiongemma_general")
    print(f"\n  functiongemma_general (base):")
    for k, v in m.items():
        print(f"    {k}: {v*100:.1f}%")

    ft_results = None
    adapter = str(args.adapter_path) if args.adapter_path.exists() and not args.base_only else None

    if adapter:
        print(f"\n{'=' * 60}")
        print(f"  FINE-TUNED: {args.base_model} + {adapter}")
        print(f"{'=' * 60}")
        ft_results = run_evaluation(
            args.base_model, adapter, tasks, args.limit, args.batch_size,
        )

        m = extract_metrics(ft_results, "functiongemma_general")
        print(f"\n  functiongemma_general (fine-tuned):")
        for k, v in m.items():
            print(f"    {k}: {v*100:.1f}%")

    generate_report(
        args.base_model, adapter,
        base_results, ft_results,
        args.limit, args.output,
    )

    bg = extract_metrics(base_results, "functiongemma_general")
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Base (general):  {bg['tool_selection_acc']*100:.1f}%")

    if ft_results:
        fg = extract_metrics(ft_results, "functiongemma_general")
        print(f"  FT (general):    {fg['tool_selection_acc']*100:.1f}% ({(fg['tool_selection_acc']-bg['tool_selection_acc'])*100:+.1f}%)")

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
