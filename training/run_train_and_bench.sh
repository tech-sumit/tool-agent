#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/workspace/bench
VENV=/workspace/venv

echo "============================================"
echo "  FunctionGemma v1 Train + Benchmark"
echo "  $(date)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
echo ""

# --- Setup ---
mkdir -p "$WORKDIR"
echo "--- Setting up venv ---"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV" --system-site-packages
fi
source "$VENV/bin/activate"
pip install --quiet lm-eval transformers peft accelerate sentencepiece protobuf safetensors trl datasets pyyaml 2>&1 | tail -3
echo "DEPS_OK"

# --- Extract ---
echo "--- Extracting package ---"
tar xzf /workspace/bench_package.tar.gz -C "$WORKDIR"
find "$WORKDIR" -name '._*' -delete 2>/dev/null || true
cd "$WORKDIR"

echo "--- Files ---"
ls -la training/data/*.jsonl
ls -la training/configs/*.yaml
echo ""

# --- Step 1: Prepare test split ---
echo "============================================"
echo "  STEP 1: Prepare test data"
echo "============================================"
python -m training.prepare_test_data \
    --input training/data/functiongemma_training_general.jsonl \
    --samples 100
echo ""

# --- Step 2: Fine-tune ---
echo "============================================"
echo "  STEP 2: Fine-tune (general data only)"
echo "============================================"
python -m training.finetune \
    --model unsloth/functiongemma-270m-it \
    --dataset training/data/functiongemma_training_general.jsonl \
    --output ./models/finetuned \
    --config training/configs/functiongemma_h100.yaml
echo ""

# --- Step 3: Benchmark ---
echo "============================================"
echo "  STEP 3: Benchmark (base vs fine-tuned)"
echo "============================================"
python -m training.benchmark \
    --base-model unsloth/functiongemma-270m-it \
    --adapter-path ./models/finetuned \
    --batch-size 16 \
    --output training/reports/benchmark_report.md
echo ""

# --- Results ---
echo "============================================"
echo "  REPORT"
echo "============================================"
cat training/reports/benchmark_report.md

echo ""
echo "============================================"
echo "  OUTPUT FILES"
echo "============================================"
ls -la models/finetuned/
ls -la training/reports/

echo ""
echo "ALL_DONE"
