.PHONY: help install test lint run evaluate-public docker-build sync-prompts sync-datasets eval-classify eval-retrieval eval-rag eval-mcp eval-e2e

UV ?= uv
RUN := $(UV) run --frozen
BASE_URL ?= http://127.0.0.1:8080
EVAL_LABEL ?= local
EVAL_ARGS ?=

help:
	@awk 'BEGIN {FS = ":.*##"}; /^[a-zA-Z_-]+:.*##/ {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime, dev and eval dependencies from uv.lock
	$(UV) sync --frozen --all-groups

test: ## Run unit and integration tests
	$(RUN) pytest

lint: ## Run static checks
	$(RUN) ruff check .

run: ## Start the API locally on port 8080
	$(RUN) uvicorn app.main:app --host 0.0.0.0 --port 8080 --no-access-log

evaluate-public: ## Run the public black-box cases against BASE_URL
	$(RUN) python scripts/evaluate_public.py --base-url $(BASE_URL)

docker-build: ## Build the local OCI image
	docker build -t aeroassist:local .

sync-prompts: ## Push changed bundled prompts to Langfuse Prompt Management
	$(RUN) python -m app.orchestration.prompt_sync

sync-datasets: ## Mirror the evaluation datasets into Langfuse Datasets
	$(RUN) --group eval python -m evaluation.langfuse_sync

eval-classify: ## Classifier route and flag accuracy for CLASSIFIER_MODEL (paid with a real backend)
	$(RUN) --group eval python -m evaluation.run classify --label $(EVAL_LABEL) $(EVAL_ARGS)

eval-retrieval: ## Retrieval recall, precision, MRR and min-score calibration (no LLM)
	$(RUN) --group eval python -m evaluation.run retrieval --label $(EVAL_LABEL) $(EVAL_ARGS)

eval-rag: ## Citation, abstention and grounding metrics (add EVAL_ARGS=--judge for paid metrics)
	$(RUN) --group eval python -m evaluation.run rag --label $(EVAL_LABEL) $(EVAL_ARGS)

eval-mcp: ## Tool selection, arguments, confirmation and write-gate metrics
	$(RUN) --group eval python -m evaluation.run mcp --label $(EVAL_LABEL) $(EVAL_ARGS)

eval-e2e: ## Public golden plus private-style cases (EVAL_ARGS=--base-url=<url> for remote)
	$(RUN) --group eval python -m evaluation.run e2e --label $(EVAL_LABEL) $(EVAL_ARGS)
