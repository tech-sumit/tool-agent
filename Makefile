.PHONY: help setup dev train data eval export serve up down status test clean publish-hf publish-github

PYTHON ?= python3
PORT ?= 8888
HOST ?= 0.0.0.0
KNOWLEDGE_DB ?= ../../n8n-templates/knowledge_db
MODEL_NAME ?= unsloth/functiongemma-270m-it
XLAM_SAMPLE ?= 10000
IRR_SAMPLE ?= 3000
OUTPUT_DIR ?= ./models/finetuned
GGUF_DIR ?= ./models/gguf

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	pip install -e ".[training]"

dev: ## Install in development mode
	pip install -e "."

data: ## Generate n8n-specific training data from knowledge_db
	$(PYTHON) -m training.generate_training_data \
		--knowledge-db $(KNOWLEDGE_DB)/node_knowledge_db.json \
		--connections $(KNOWLEDGE_DB)/connection_patterns.json \
		--output training/data/training_data.jsonl

download: ## Download external datasets (xlam-60k + irrelevance-7.5k) and merge with n8n data
	$(PYTHON) -m training.download_datasets \
		--output-dir training/data \
		--n8n-data training/data/training_data.jsonl \
		--xlam-sample $(XLAM_SAMPLE) \
		--irrelevance-sample $(IRR_SAMPLE)

convert: ## Convert combined data to FunctionGemma format
	$(PYTHON) -m training.convert_to_functiongemma \
		--input training/data/combined_training_data.jsonl \
		--output training/data/functiongemma_training.jsonl

train: ## Fine-tune FunctionGemma on converted dataset
	$(PYTHON) -m training.finetune \
		--model $(MODEL_NAME) \
		--dataset training/data/functiongemma_training.jsonl \
		--output $(OUTPUT_DIR) \
		--config training/configs/functiongemma.yaml

eval: ## Evaluate model accuracy on test set
	$(PYTHON) -m training.evaluate \
		--dataset training/data/training_data.jsonl

benchmark: ## Run full benchmark: base FunctionGemma vs fine-tuned
	$(PYTHON) -m training.benchmark

export: ## Export fine-tuned model to GGUF for Ollama
	$(PYTHON) -m training.export_gguf \
		--model $(OUTPUT_DIR) \
		--output $(GGUF_DIR)

serve: ## Start the agent server (foreground)
	$(PYTHON) -m agent.server --host $(HOST) --port $(PORT)

up: ## Start agent server (background via Docker)
	docker compose -f docker/docker-compose.yml up -d

down: ## Stop agent server
	docker compose -f docker/docker-compose.yml down

status: ## Check agent status and registered tools
	@curl -sf http://localhost:$(PORT)/health | python3 -m json.tool 2>/dev/null || echo "Agent not running"
	@echo "---"
	@curl -sf http://localhost:$(PORT)/tools | python3 -m json.tool 2>/dev/null || true

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

publish-hf: ## Publish fine-tuned model to Hugging Face Hub
	$(PYTHON) -m training.publish_hf \
		--model $(OUTPUT_DIR) \
		--repo sumitagrawal/functiongemma-270m-tool-agent \
		--dataset training/data/functiongemma_training_general.jsonl \
		--base-model $(MODEL_NAME)

clean: ## Remove generated artifacts
	rm -rf models/ training/data/*.jsonl __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
