"""Prepare test data splits for lm-evaluation-harness benchmarking.

Creates a held-out general test set from the training data:
  - general_test.jsonl: stratified sample from the FunctionGemma dataset

Usage:
    python -m training.prepare_test_data
    python -m training.prepare_test_data --input training/data/functiongemma_training_general.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples)} examples to {path}")


def stratified_sample(examples: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n examples with proportional category representation."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for ex in examples:
        by_cat[ex.get("category", "unknown")].append(ex)

    total = len(examples)
    sampled = []
    for cat, cat_examples in sorted(by_cat.items()):
        cat_n = max(1, round(len(cat_examples) / total * n))
        rng.shuffle(cat_examples)
        sampled.extend(cat_examples[:cat_n])

    rng.shuffle(sampled)
    return sampled[:n]


def main():
    parser = argparse.ArgumentParser(description="Prepare test data for benchmarking")
    parser.add_argument(
        "--input", type=Path,
        default=Path("training/data/functiongemma_training_general.jsonl"),
        help="Training data JSONL to sample from",
    )
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    data_dir = Path("training/data")

    print(f"Loading training data from {args.input}...")
    fg_data = load_jsonl(args.input)
    print(f"  {len(fg_data)} total examples")

    cats = defaultdict(int)
    for ex in fg_data:
        cats[ex.get("category", "unknown")] += 1
    print("  Categories:")
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    print(f"\nCreating general test set ({args.samples} stratified samples)...")
    general_test = stratified_sample(fg_data, args.samples)
    save_jsonl(general_test, data_dir / "general_test.jsonl")

    print("\nTest data ready for lm-evaluation-harness.")


if __name__ == "__main__":
    main()
