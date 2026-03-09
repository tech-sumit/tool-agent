"""Export fine-tuned FunctionGemma to GGUF format for Ollama deployment.

Merges the LoRA adapter with the base model using PEFT, then converts
to GGUF using llama.cpp's conversion script.

Usage:
    python -m training.export_gguf \
        --model ./models/finetuned \
        --output ./models/gguf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"


def merge_adapter(adapter_path: Path, merged_path: Path):
    """Merge LoRA adapter with base model using PEFT."""
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(adapter_path / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg["base_model_name_or_path"]

    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved ({sum(f.stat().st_size for f in merged_path.glob('*')) / 1e6:.1f} MB)")


def ensure_llama_cpp() -> Path:
    """Clone llama.cpp if not already present, return path."""
    llama_dir = Path.home() / ".cache" / "llama.cpp"
    convert_script = llama_dir / "convert_hf_to_gguf.py"

    if convert_script.exists():
        print(f"Using existing llama.cpp at {llama_dir}")
        return convert_script

    print(f"Cloning llama.cpp to {llama_dir}...")
    subprocess.run(
        ["git", "clone", "--depth=1", LLAMA_CPP_REPO, str(llama_dir)],
        check=True,
    )

    req_file = llama_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if req_file.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gguf", "numpy", "sentencepiece", "protobuf"],
            check=True,
        )

    return convert_script


def convert_to_gguf(merged_path: Path, output_path: Path, quantization: str):
    """Convert merged HF model to GGUF format."""
    convert_script = ensure_llama_cpp()
    output_path.mkdir(parents=True, exist_ok=True)

    gguf_file = output_path / "functiongemma-270m-tool-agent-f32.gguf"
    print(f"Converting to GGUF: {gguf_file}")
    subprocess.run(
        [
            sys.executable, str(convert_script),
            str(merged_path),
            "--outfile", str(gguf_file),
            "--outtype", "f32",
        ],
        check=True,
    )

    if quantization and quantization != "f32":
        llama_dir = convert_script.parent
        quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = shutil.which("llama-quantize")

        if quantize_bin:
            quantized = output_path / f"functiongemma-270m-tool-agent-{quantization}.gguf"
            print(f"Quantizing to {quantization}: {quantized}")
            subprocess.run(
                [str(quantize_bin), str(gguf_file), str(quantized), quantization.upper()],
                check=True,
            )
            gguf_file.unlink()
            gguf_file = quantized
        else:
            print(f"llama-quantize not found, keeping f32. Build llama.cpp or install llama-quantize to quantize.")

    print(f"GGUF model: {gguf_file} ({gguf_file.stat().st_size / 1e6:.1f} MB)")
    return gguf_file


def create_modelfile(output_path: Path, model_name: str):
    """Create an Ollama Modelfile for the exported GGUF."""
    gguf_files = list(output_path.glob("*.gguf"))
    if not gguf_files:
        print("No GGUF file found, skipping Modelfile creation.")
        return

    gguf_file = gguf_files[0]
    modelfile_content = f"""FROM {gguf_file.name}

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER num_ctx 2048

TEMPLATE \"\"\"{{{{- if .System }}}}<start_of_turn>developer
{{{{ .System }}}}<end_of_turn>
{{{{- end }}}}
<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
\"\"\"

SYSTEM "You are a function calling assistant. Given a user query and available tools, select the appropriate tool and generate the function call."
"""

    modelfile_path = output_path / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    print(f"Ollama Modelfile written to {modelfile_path}")
    print(f"\nTo import into Ollama:")
    print(f"  cd {output_path}")
    print(f"  ollama create {model_name} -f Modelfile")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF")
    parser.add_argument("--model", required=True, type=Path, help="Fine-tuned adapter path")
    parser.add_argument("--output", required=True, type=Path, help="GGUF output directory")
    parser.add_argument("--quantization", default="q4_k_m", help="Quantization method (f32, q4_k_m, q8_0)")
    parser.add_argument("--ollama-name", default="tool-agent", help="Ollama model name")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge (model is already merged)")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: {args.model} does not exist", file=sys.stderr)
        sys.exit(1)

    merged_path = args.output / "merged"

    if not args.skip_merge:
        merge_adapter(args.model, merged_path)
    else:
        merged_path = args.model

    convert_to_gguf(merged_path, args.output, args.quantization)
    create_modelfile(args.output, args.ollama_name)

    if not args.skip_merge and merged_path.exists():
        print(f"\nCleaning up merged model at {merged_path}...")
        shutil.rmtree(merged_path)

    print("\nDone! Next steps:")
    print(f"  cd {args.output}")
    print(f"  ollama create {args.ollama_name} -f Modelfile")
    print(f"  ollama run {args.ollama_name}")


if __name__ == "__main__":
    main()
