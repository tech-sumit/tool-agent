"""Fine-tune language models for function calling on n8n integration data.

Supports two model formats:
  - gemma: FunctionGemma (google/functiongemma-270m-it) — Gemma 3 turn markers
  - chatml: xLAM-2 / Qwen2 models — ChatML template

Works on Apple Silicon MPS, CUDA GPU, or CPU via PEFT + TRL.

Usage:
    python -m training.finetune \
        --model google/functiongemma-270m-it \
        --dataset training/data/functiongemma_training.jsonl \
        --output ./models/finetuned \
        --config training/configs/functiongemma.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            print("  Device: CUDA GPU (bf16)")
            return "cuda", torch.bfloat16
        print("  Device: CUDA GPU (fp16)")
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Device: Apple Silicon MPS")
        return "mps", torch.float32
    print("  Device: CPU (training will be slow)")
    return "cpu", torch.float32


def format_chatml(messages: list[dict]) -> str:
    """Format messages into ChatML template for xLAM-2 (Qwen2)."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def detect_format(raw_examples: list[dict]) -> str:
    """Auto-detect data format from content."""
    sample = raw_examples[0]
    if "text" in sample:
        text = sample["text"]
        if "<start_of_turn>" in text:
            return "gemma"
        if "<|im_start|>" in text:
            return "chatml"
        return "preformatted"
    if "messages" in sample:
        return "chatml"
    return "unknown"


def train_peft(args, cfg, raw_examples):
    """PEFT + TRL training. Works on MPS, CUDA, and CPU."""
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    max_seq_length = train_cfg.get("max_seq_length", 2048)

    data_format = detect_format(raw_examples)
    print(f"  Data format: {data_format}")

    device, dtype = detect_device()
    use_fp16 = dtype == torch.float16
    use_bf16 = dtype == torch.bfloat16

    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device} if device != "cpu" else "auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
    )

    print("Applying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if device == "mps":
        model.enable_input_require_grads()

    def apply_template(example):
        if "text" in example and example["text"]:
            return {"text": example["text"]}
        if "messages" in example:
            return {"text": format_chatml(example["messages"])}
        raise ValueError(f"Example has neither 'text' nor 'messages': {list(example.keys())}")

    print("Preparing dataset...")
    dataset = Dataset.from_list(raw_examples)
    dataset = dataset.map(apply_template, remove_columns=dataset.column_names)

    test_size = train_cfg.get("test_size", 0.1)
    split = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")

    batch_size = train_cfg.get("per_device_train_batch_size", 4)
    if device == "mps" and "functiongemma" not in args.model.lower():
        batch_size = min(batch_size, 2)

    output_dir = str(args.output)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_length=max_seq_length,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=train_cfg.get("logging_steps", 50),
        save_steps=train_cfg.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 250),
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        dataset_text_field="text",
        dataloader_pin_memory=False if device == "mps" else True,
        use_cpu=device == "cpu",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=training_args,
        peft_config=None,
    )

    print(f"\nStarting training on {device.upper()}...")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {batch_size} (x{training_args.gradient_accumulation_steps} accum)")
    print(f"  Effective batch: {batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Training examples: {len(split['train'])}\n")

    trainer.train()

    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune models for function calling")
    parser.add_argument(
        "--model",
        default="google/functiongemma-270m-it",
        help="Base model (default: google/functiongemma-270m-it)",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Training data JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--config", type=Path, default=Path("training/configs/functiongemma.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Loading training data from {args.dataset}...")
    raw_examples = load_dataset_from_jsonl(args.dataset)
    print(f"  {len(raw_examples)} examples loaded")

    train_peft(args, cfg, raw_examples)
    print("\nFine-tuning complete.")


if __name__ == "__main__":
    main()
