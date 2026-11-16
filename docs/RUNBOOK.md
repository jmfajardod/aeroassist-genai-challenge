# Runbook

This document explains how to install, run, test, evaluate, deploy, roll back and clean up AeroAssist. The commands assume the repository root as the working directory.

The deployment runs in one GCP project, in us-central1, as the Cloud Run service aeroassist. The commands below use two shell variables:

```bash
export PROJECT_ID=your-gcp-project-id
export SERVICE_URL=$(gcloud run services describe aeroassist --region us-central1 --project "$PROJECT_ID" --format='value(status.url)')
```

## Local setup

Requirements: uv, Docker, and Python 3.12 (uv downloads it if it is missing).

1. Install the dependencies: `make install`
2. Create the configuration: `cp .env.example .env`
3. Fill only what you need. Without changes the app runs fully offline with the fake LLM and an in-process index.

Main settings in .env:

- LLM_BACKEND: fake (default, no key) or openai.
- OPENAI_API_KEY: required when LLM_BACKEND=openai.
- CLASSIFIER_MODEL, TOOLS_MODEL, ANSWER_MODEL and their *_MAX_TOKENS and *_REASONING_EFFORT: model per role.
- LLM_BASE_URL: provider endpoint or region (empty uses the default).
- QDRANT_URL: empty builds the index in process at startup; set it to use a Qdrant server.
- EMBEDDING_CACHE_DIR: folder for the embedding models (the image uses /app/models).
- RAG_TOP_K, RAG_MIN_SCORE and RAG_EVIDENCE_SCORE: retrieval and abstention thresholds.
- LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL: tracing (off when the keys are empty).
- MAX_CONCURRENT_REQUESTS and RATE_LIMIT_PER_MINUTE: local limits (429 when exceeded).

## Run locally

- API: `make run`, then open http://127.0.0.1:8080/health
- With a Qdrant server: `docker compose up -d qdrant`, set `QDRANT_URL=http://localhost:6333` in .env, run `python -m app.retrieval.ingest` once, and then `make run`.
- Docker image: `make docker-build`, then `docker run --rm -p 8080:8080 --env-file .env aeroassist:local`

Quick checks:

```bash
curl -s localhost:8080/health
curl -s localhost:8080/ready
curl -s -X POST localhost:8080/v1/chat -H 'content-type: application/json' \
  -d '{"message":"¿Cuál es el estado de LA401 para el 2026-08-06?"}'
```

## Tests and evaluation

- Lint and tests (no paid calls, CI runs the same): `make lint test`
- Public golden set against any URL: `make evaluate-public BASE_URL="$SERVICE_URL"`

The following evaluations use real models when LLM_BACKEND=openai, so they cost money (a few cents per run). The reports are written to docs/test_results/<suite>-<label>.json:

- Retrieval only, without LLM: `make eval-retrieval EVAL_LABEL=name`
- Classifier: `make eval-classify EVAL_LABEL=name`
- RAG with the Ragas judge: `EVAL_JUDGE_MODEL=gpt-6.1-terra make eval-rag EVAL_LABEL=name EVAL_ARGS=--judge`
- MCP tool calling: `make eval-mcp EVAL_LABEL=name`
- End to end, locally or remote: `make eval-e2e EVAL_LABEL=name EVAL_ARGS=--base-url=https://...`

The label local is ignored by git, so use another label only for runs that should be kept as evidence.

## Knowledge base index

- Build: the documents in knowledge_base/ are split by section, embedded (MiniLM and BM25) and stored in Qdrant.
- Version: a hash of the document contents. /ready shows index=false and returns 503 if the stored version does not match the knowledge base of the running image.
- Refresh in the deployed service: the index is built in process at startup, so a new release with the changed documents refreshes it.
- Refresh with a Qdrant server: `QDRANT_URL=... QDRANT_API_KEY=... python -m app.retrieval.ingest`. It is idempotent: it updates changed documents, skips unchanged ones and deletes documents that were removed.

## Prompts and datasets in Langfuse

- Push the bundled prompts as new versions (only the ones that changed): `make sync-prompts`
- Serve the prompts from Langfuse: set `PROMPT_SOURCE=langfuse` (the bundled copy is the fallback).
- Mirror the evaluation datasets: `make sync-datasets`

## Deploy

### First time in a new project

These steps were already done for the current project. To repeat them in another project:

1. Bootstrap (APIs, state bucket, Workload Identity Federation, Terraform and plan accounts, budget):

   ```bash
   PROJECT_ID=my-project GITHUB_REPO=owner/repo GITHUB_ACTOR=owner \
   BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX BUDGET_AMOUNT=80000COP ./infra/bootstrap.sh
   ```

   The budget amount must use the currency of the billing account.
2. Set the GitHub repository variables that the script prints at the end.
3. First Terraform apply, without the service:

   ```bash
   cd infra/terraform/projects/dev
   terraform init -backend-config="bucket=my-project-aeroassist-tfstate"
   TF_VAR_project_id=my-project TF_VAR_github_repository=owner/repo TF_VAR_image="" terraform apply
   ```

4. Add the secret values (the values never go in Terraform):

   ```bash
   printf '%s' "$OPENAI_API_KEY" | gcloud secrets versions add aeroassist-openai-api-key --data-file=-
   printf '%s' "$LANGFUSE_PUBLIC_KEY" | gcloud secrets versions add aeroassist-langfuse-public-key --data-file=-
   printf '%s' "$LANGFUSE_SECRET_KEY" | gcloud secrets versions add aeroassist-langfuse-secret-key --data-file=-
   ```

5. Push a first image, apply again with its digest (this creates the service), and save the digest in the INITIAL_IMAGE repository variable.

### Normal release

1. Merge the changes to main (ci runs lint, tests and the Docker build).
2. Create a release branch: `git checkout -b release/1.0.2 main && git push -u origin release/1.0.2`
3. Only if the infrastructure changed: run terraform-apply on the release branch from the Actions tab, and approve the production environment.
4. Run cd on the release branch and approve the production environment. It runs lint and tests, builds and pushes the image, deploys it by digest with all the traffic, and runs the smoke test and the public evaluation.
5. The job summary shows the commit, image digest, revision, URL and evaluation result.

Only the repository owner can run terraform-apply and cd, only from release/* branches, and only after approval.

## Rollback

1. List the revisions:

   ```bash
   gcloud run revisions list --service aeroassist --region us-central1 --project "$PROJECT_ID"
   ```

2. Send all the traffic to the previous revision:

   ```bash
   gcloud run services update-traffic aeroassist --region us-central1 --project "$PROJECT_ID" \
     --to-revisions=PREVIOUS_REVISION=100
   ```

3. Check: `curl -s "$SERVICE_URL/ready"` and `make evaluate-public BASE_URL="$SERVICE_URL"`

The next cd run moves the traffic to the new revision again. It is also possible to run cd from the previous release branch, which builds that code again.

## Rotate a secret

1. Add a new version: `printf '%s' "$NEW_VALUE" | gcloud secrets versions add aeroassist-openai-api-key --data-file=-`
2. Deploy a new revision so it reads the latest version (run cd, or `gcloud run services update aeroassist --region us-central1 --update-labels=rotated=$(date +%s)`).
3. Disable the old version: `gcloud secrets versions disable OLD_VERSION --secret=aeroassist-openai-api-key`

## Cleanup

Quick, only the service:

```bash
gcloud run services delete aeroassist --region us-central1 --project "$PROJECT_ID"
```

Full, only the resources created for AeroAssist and by name (never delete the project):

1. Terraform resources (service, registry and images, secrets, runtime and deployer accounts):

   ```bash
   cd infra/terraform/projects/dev
   terraform init -backend-config="bucket=${PROJECT_ID}-aeroassist-tfstate"
   TF_VAR_project_id="$PROJECT_ID" TF_VAR_github_repository=OWNER/REPO \
   TF_VAR_image="$(gh variable get INITIAL_IMAGE -R OWNER/REPO)" terraform destroy
   ```

2. Bootstrap resources:

   ```bash
   gcloud iam workload-identity-pools delete github --location=global --project "$PROJECT_ID"
   gcloud iam service-accounts delete aeroassist-terraform@${PROJECT_ID}.iam.gserviceaccount.com --project "$PROJECT_ID"
   gcloud iam service-accounts delete aeroassist-plan@${PROJECT_ID}.iam.gserviceaccount.com --project "$PROJECT_ID"
   gcloud billing budgets list --billing-account=BILLING_ACCOUNT
   gcloud billing budgets delete BUDGET_ID --billing-account=BILLING_ACCOUNT
   gcloud storage rm --recursive gs://${PROJECT_ID}-aeroassist-tfstate
   ```

3. Optional: delete the GitHub repository variables and the production environment.

## Troubleshooting

- /ready returns 503 with index=false: the stored index does not match the knowledge base of the image. With a Qdrant server, run the ingest CLI; in Cloud Run, check the startup logs.
- /ready returns 503 with llm=false: OPENAI_API_KEY is missing or the secret has no version.
- /ready returns 503 with mcp=false: the MCP subprocess did not start; check the logs for the mcp_server errors.
- 429 responses: the local limit was reached (10 concurrent requests or 60 per minute). This is expected under bursts, since the service runs with max 1 instance.
- 502 or 504 responses: the LLM provider or a tool failed after the retries, or the request went over the 35 s deadline. The request ID is in the X-Request-ID header and in the logs.
- The revision does not start and shows "Memory limit exceeded": the index and the models need about 1 GiB at startup, so keep the service at 2 GiB.
- GitHub Actions fails with "must specify exactly one of workload_identity_provider": the repository variables are missing.
- terraform-apply or cd is skipped or cannot authenticate: they only run from release/* branches and for the repository owner.
- Langfuse returns 401: check that LANGFUSE_BASE_URL matches the region of the keys.
- Logs: `gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="aeroassist"' --project "$PROJECT_ID" --limit 50`
