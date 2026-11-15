.PHONY: help install test lint run evaluate-public docker-build eval-rag eval-mcp eval-e2e

UV ?= uv
RUN := $(UV) run --frozen
BASE_URL ?= http://127.0.0.1:8080

help:
	@awk 'BEGIN {FS = ":.*##"}; /^[a-zA-Z_-]+:.*##/ {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime, dev and eval dependencies from uv.lock
	$(UV) sync --frozen --all-groups

test: ## Run unit and integration tests
	$(RUN) pytest

lint: ## Run static checks
	$(RUN) ruff check .

run: ## Start the API locally on port 8080
	$(RUN) uvicorn app.main:app --host 0.0.0.0 --port 8080

evaluate-public: ## Run the public black-box cases against BASE_URL
	$(RUN) python scripts/evaluate_public.py --base-url $(BASE_URL)

docker-build: ## Build the local OCI image
	docker build -t aeroassist:local .

eval-rag: ## Retrieval and grounding evaluation (paid LLM calls)
	@echo "eval-rag is not implemented yet (roadmap commit 4)" && exit 1

eval-mcp: ## Tool selection and argument evaluation (paid LLM calls)
	@echo "eval-mcp is not implemented yet (roadmap commit 4)" && exit 1

eval-e2e: ## End-to-end evaluation against BASE_URL
	@echo "eval-e2e is not implemented yet (roadmap commit 4)" && exit 1
