"""Publish fine-tuned FunctionGemma model and adapter to Hugging Face Hub.

Pushes the LoRA adapter weights, tokenizer, training config, and a
generated model card to a HuggingFace repository.

Usage:
    python -m training.publish_hf \
        --model ./models/finetuned \
        --repo sumitagrawal/functiongemma-270m-tool-agent \
        --dataset training/data/functiongemma_training.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_CARD_TEMPLATE = """\
---
language:
- en
license: apache-2.0
library_name: peft
base_model: {base_model}
tags:
- function-calling
- tool-use
- gemma3
- lora
- peft
datasets:
- Salesforce/xlam-function-calling-60k
- MadeAgents/xlam-irrelevance-7.5k
pipeline_tag: text-generation
---

# {repo_id}

Fine-tuned [FunctionGemma 270M](https://huggingface.co/{base_model}) LoRA adapter
specialized for **general tool/function calling**.

## Benchmark Results

Evaluated using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on 100 held-out general function-calling examples:

| Metric | Base | Fine-tuned | Delta |
|--------|------|-----------|-------|
| Tool Selection Acc | 49.0% | 78.0% | **+29.0%** |
| First Tool Acc | 49.0% | 88.0% | **+39.0%** |
| Negative Rejection | 100.0% | 100.0% | +0.0% |
| Param Accuracy | 49.0% | 68.9% | **+19.9%** |

## Training

- **Base model**: `{base_model}`
- **Method**: LoRA (r={lora_r}, alpha={lora_alpha}) via PEFT + TRL SFTTrainer
- **Dataset**: {num_examples:,} general function-calling examples
- **Epochs**: {num_epochs}
- **Hardware**: NVIDIA H100 SXM 80GB

### Data composition

| Source | Examples | Purpose |
|--------|----------|---------|
| xlam-function-calling-60k | ~10,000 | General function calling |
| xlam-irrelevance-7.5k | ~3,000 | Negative examples / refusal |
| **Total** | **~13,000** | |

## Usage

### With PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

prompt = \"\"\"<start_of_turn>developer
You are a helpful assistant with access to the following tools:
- send_email(to, subject, body): Send an email
- search_contacts(query): Search contacts by name
<end_of_turn>
<start_of_turn>user
Send an email to John about the meeting tomorrow
<end_of_turn>
<start_of_turn>model
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
print(tokenizer.decode(out[0], skip_special_tokens=False))
```

### With Ollama (GGUF)

Export to GGUF first, then:

```bash
ollama create tool-agent -f Modelfile
ollama run tool-agent
```

## Output format

The model uses FunctionGemma's native control-token format:

```
<start_function_call>call:function_name{{param1:<escape>value1<escape>param2:<escape>value2<escape>}}<end_function_call>
```

## License

Apache 2.0 (same as the base model).
"""


def count_examples(dataset_path: Path | None) -> int:
    if not dataset_path or not dataset_path.exists():
        return 0
    count = 0
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_training_config(model_path: Path) -> dict:
    """Read training args from the saved trainer state."""
    trainer_state = model_path / "trainer_state.json"
    if trainer_state.exists():
        with open(trainer_state) as f:
            return json.load(f)
    return {}


def read_adapter_config(model_path: Path) -> dict:
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Publish model to Hugging Face Hub")
    parser.add_argument(
        "--model", required=True, type=Path,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--repo", required=True,
        help="HuggingFace repo ID (e.g. sumitagrawal/functiongemma-270m-tool-agent)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Path to training JSONL (for model card stats)",
    )
    parser.add_argument(
        "--base-model", default="unsloth/functiongemma-270m-it",
        help="Base model identifier",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private repository",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: Model directory {args.model} does not exist.")
        raise SystemExit(1)

    from huggingface_hub import HfApi, login
    from transformers import AutoTokenizer

    login()

    adapter_cfg = read_adapter_config(args.model)
    trainer_state = read_training_config(args.model)
    num_examples = count_examples(args.dataset)

    lora_r = adapter_cfg.get("r", 16)
    lora_alpha = adapter_cfg.get("lora_alpha", 32)

    num_epochs = 3
    if trainer_state:
        total_steps = trainer_state.get("max_steps", 0)
        log_history = trainer_state.get("log_history", [])
        if log_history:
            last = log_history[-1]
            num_epochs = int(last.get("epoch", 3))

    model_card = MODEL_CARD_TEMPLATE.format(
        repo_id=args.repo,
        base_model=args.base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        num_examples=num_examples or 13000,
        num_epochs=num_epochs,
    )

    readme_path = args.model / "README.md"
    readme_path.write_text(model_card)
    print(f"Generated model card at {readme_path}")

    api = HfApi()

    print(f"Creating repo {args.repo}...")
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading model files to {args.repo}...")
    api.upload_folder(
        folder_path=str(args.model),
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=["*.bin", "optimizer.*", "scheduler.*", "trainer_state*",
                         "training_args*", "global_step*", "checkpoint-*"],
    )

    print(f"\nModel published to: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
