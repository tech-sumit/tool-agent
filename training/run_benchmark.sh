#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/workspace/bench
VENV=/workspace/venv

echo "=== GPU Benchmark Runner ==="
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

mkdir -p "$WORKDIR"

echo "--- Setting up venv ---"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV" --system-site-packages
fi
source "$VENV/bin/activate"
pip install --quiet lm-eval transformers peft accelerate sentencepiece protobuf safetensors 2>&1 | tail -3

echo "--- Extracting package ---"
tar xzf /workspace/bench_package.tar.gz -C "$WORKDIR"
find "$WORKDIR" -name '._*' -delete 2>/dev/null || true

echo "--- Verifying files ---"
ls -la "$WORKDIR/training/data/"*.jsonl
ls -la "$WORKDIR/training/tasks/functiongemma_eval/"*.yaml
ls -la "$WORKDIR/models/finetuned/adapter_config.json"

echo ""
echo "=== Starting Benchmark ==="
cd "$WORKDIR"
python -m training.benchmark \
    --base-model unsloth/functiongemma-270m-it \
    --adapter-path ./models/finetuned \
    --batch-size 16 \
    --output training/reports/benchmark_report.md

echo ""
echo "=== Benchmark Complete ==="
cat training/reports/benchmark_report.md
echo ""
echo "BENCHMARK_DONE"
