# Technical decisions

## Highest completed level

- Claimed level: Level 4.
- What is demonstrably working:
  - Live service on Cloud Run deployed by the cd workflow from a release branch.
  - Public golden set: 8/8 against the live URL (inside the CD smoke test), and also local and in Docker.
- What is incomplete:
  - Qdrant runs in process inside the container instead of on its own server.

## Architecture

- Main components:
  The service is a FastAPI app with a LangGraph workflow behind it. For every POST /v1/chat the following flow is executed:
  1. Middleware: assigns the request ID, applies the concurrency and rate limits, and validates the body with pydantic.
  2. Classify: an LLM call with structured output decides the route (help, unsupported, injection or answer) and whether the question needs live data, policy or a write. Injection returns 400, and help or unsupported return a fixed answer without more LLM calls.
  3. Fan out: with Send of LangGraph, the tools branch and the retrieval branch run in parallel.
  4. Write requests skip the LLM and go to the write gate, deterministic code that is the only path to create_reaccommodation_hold.
  5. Answer: an LLM call with structured output writes the answer in Spanish and lists the chunk IDs it used.
  6. Validate: only cited chunks that were actually retrieved become citations, and an ungrounded answer abstains.

- Why this LLM/provider abstraction:
  The graph only knows the LLMProvider interface (app/llm.py), which has two implementations: LangChain chat models (OpenAI by default, other providers by configuration) and a deterministic fake backend, so tests and CI never call a paid model. Each role has its own model, token cap and reasoning effort in environment variables, and the models were chosen with evaluations: gpt-6.1-terra for the classifier (same accuracy as gpt-6-sol at half the price), gpt-6-luna for tools and gpt-6.1-terra for answers (best faithfulness).
  Design principle: the LLM interprets the question and writes the answer; the code validates the arguments, decides the writes and builds the citations.
- Why this tool orchestration approach:
  A defined graph with bounded loops instead of a free ReAct agent, since it is easier to test, explain and bound in latency and cost. Inside the tools branch the LLM uses native tool calling over the provided MCP server, but every proposed call goes through code before it runs:
  - Allowlist of the three read tools. The write tool is never offered to the LLM.
  - Strict pydantic schemas (formats of flight, booking, IATA code and date, and max_results capped at 5).
  - Provenance: identifiers and dates must come from the user message, from a relative date resolved against the reference date, or from an earlier tool result. This is why the assistant asks for the date instead of guessing it.

  Measured: tool call accuracy and F1 of 1.0 on the MCP dataset, so the fallback approach (the LLM extracts entities and the code picks the tool) was not needed.
- Why this retrieval/index strategy:
  Qdrant with FastEmbed hybrid search: a multilingual dense model (paraphrase-multilingual-MiniLM-L12-v2) plus BM25, fused with RRF. The documents are split by section with the LangChain splitters, with overlap for long sections. I compared five FastEmbed models on the RAG dataset; all reached recall@3 of 1.0, and MiniLM was the best balance between quality, size (0.22 GB), latency (10 to 26 ms per query) and memory. jina-v2-base-es ranked a bit better, but it is about 3 times bigger.
  - Thresholds on the cosine score: an evidence gate of 0.45 on the best chunk decides when to abstain, and a floor of 0.20 per chunk keeps the BM25 hits.
  - Index lifecycle: the version is a hash of the document contents. The ingest upserts changed documents and deletes removed ones, and /ready returns 503 if the stored version does not match the knowledge base in the image.
  - Local vs managed: for 7 documents a local index is cheaper, so the deployed service builds the index in process at startup (about one second) and no extra server is needed. The same code works against a Qdrant server (docker compose locally), which is the option for a bigger corpus or more than one instance.

## Evaluation

- Metrics used:
  - Retrieval, without LLM: recall@k, precision@k, hit@1 and MRR by doc_id, and the score gap used to set the thresholds.
  - RAG answers: citation precision and recall, irrelevant citation rate, abstain and qualify accuracy, and Ragas Faithfulness and AnswerRelevancy with gpt-6.1-terra as judge.
  - Tools (MCP): Ragas ToolCallAccuracy and ToolCallF1, required tools, confirmation accuracy and write-gate violations.
  - Classifier: route accuracy, injection recall and false injection rate.
  - End to end: the public golden checks, latency p50 and p95, and cost per request.

  The datasets are separate files and run with make eval-retrieval, eval-classify, eval-rag, eval-mcp and eval-e2e.
- Baseline and final results on the public golden set:

  | Revision | Public golden |
  |---|---|
  | Starter code | 0/8 (/v1/chat returned 501) |
  | Classifier graph, without tools or RAG | 3/8 |
  | MCP tools and write gate | 7/8 (only the baggage citation failed) |
  | RAG, local and Docker | 8/8 |
  | Live Cloud Run, deployed by CD | 8/8 |
- One failure you investigated and what changed:
  In the first run with real models, the classifier sent every "Confirmo …" message to prompt injection (13.5% false injections with gpt-6-luna), so the confirmed hold of public case 07 would have returned 400. The reason was that the classifier did not receive confirm_action, and the prompt treated "claims a confirmation" as an attack. I passed confirm_action to the classifier and stated that asking about the assistant limits is a policy question, not an attack; after that, the route accuracy was 1.0 with gpt-6.1-terra. The same run showed a second problem: with max_tokens=256 the reasoning tokens used the whole budget and no JSON came back, so the caps went to 1024 and 2048.

## Reliability, security and cost

- Timeout/retry/fallback policy:
  - Each LLM call: 8 s timeout and up to 3 retries with 1 s of backoff, only for retryable errors (timeouts, connection errors, throttling and 5xx).
  - Each MCP call: 5 s timeout, and the session restarts after any failure.
  - Whole request: 35 s deadline (the evaluator waits 40 s), which cancels any pending retry.
  - Status mapping: provider or tool failure returns 502, timeout returns 504, a dependency not ready returns 503 and the local limit returns 429.
  - Loop caps: 3 tool rounds and 6 tool calls per request. There is no backup LLM for now.
- Confirmation and idempotency policy:
  The write gate is the only code path to create_reaccommodation_hold, and it does not depend on the LLM:
  1. The booking, the target flight and the key come only from the current message.
  2. The booking and the target are checked with read tools: same origin and destination, and within ±3 days.
  3. Without confirm_action=true it only runs reads and returns requires_confirmation=true, mentioning special assistance (for example WCHR) if the booking has it.
  4. With confirmation it creates at most one hold, with the client key or a derived one (auto- plus a sha256 of booking, flight and date), so a retry reuses the same hold.

  A confirmed request without a booking or a flight returns 400. A booking that is not found, or a flight that is not a valid alternative, returns 200 with the explanation and the valid options. Writes that did not run never appear in tool_calls. Measured: 0 write-gate violations in every evaluation.
- Prompt-injection/data-exfiltration controls:
  - The classifier detects injection in any language and the request returns 400 (injection recall of 1.0 on the labeled cases).
  - Messages, retrieved text and tool results are treated as data, and the prompts state that they cannot change the rules.
  - Tool arguments are validated by schema, allowlist and provenance before they run.
  - Responses never include prompts, reasoning, credentials or extra passenger data.
- Logging and PII policy:
  Logs are JSON with request ID, route, status and latency, and they never include the message. One redaction module removes emails, phones, cards, RUT, DNI, passports, IPs, URLs and booking codes from the logs and from the Langfuse traces (through the mask hook, which fails closed). The text sent to the LLM only loses the personal data, since the booking codes and flight numbers are needed for the tools. I checked a real trace through the Langfuse API, and no booking code or email was stored. Person names are a known gap.
- Expected cost drivers and limits:
  - LLM tokens are the main cost. Measured: about 0.0013 USD per request, 0.0016 per tool request and 0.0019 per policy request. Help, unsupported, injection and holds do not call the answer model.
  - Limits: token caps per role, reasoning effort none for tools and answers (low for the classifier), loop caps, 10 concurrent requests and 60 per minute, and max 1 Cloud Run instance.
  - Cloud Run runs with min instances 1 during the review, since a cold start with min instances 0 took 38.5 s (the evaluator waits 40 s). With CPU billed only during requests, the idle instance costs roughly 15 to 20 USD per month; after the review it goes back to 0.
  - Budget alert of about 20 USD, with alerts at 50%, 90% and 100%.
  - Retrieval runs locally with ONNX models, so it does not use tokens. There is no response cache for now, since most questions depend on live data.

## Cloud delivery

- Platform and region:
  GCP Cloud Run in us-central1 with 1 vCPU, 2 GiB (1 GiB ran out of memory while building the index), min 1 and max 1 instances (min 0 had a cold start of 38.5 s: about 22 s to start the container and 16 s of startup probes), concurrency 10, 80 s timeout, a startup probe on /ready and a liveness probe on /health. The embedding models are baked into the image, so the container starts without network access.
- Authentication from CI/CD:
  GitHub Actions with Workload Identity Federation, without JSON keys. The provider only accepts this repository and my GitHub user, and the Terraform and deployer accounts only accept runs from release/* branches. The workflows are:
  - ci: lint, tests and Docker build on every push and pull request.
  - terraform-plan: plan on pull requests, with a read-only account.
  - terraform-apply and cd: manual, only from release/* branches, only for the repository owner and after my approval in the production environment. cd runs lint and tests, builds and pushes the image, deploys it by digest with all the traffic, and runs the smoke test and the public evaluation against the live URL.
- Runtime identity:
  A dedicated service account (aeroassist-runtime) that can only read its three secrets in Secret Manager (the OpenAI and Langfuse keys), which are mounted as environment variables. The deployer can only push images to one repository, deploy one service and act as the runtime account. The infrastructure is Terraform (infra/terraform), with the state in a versioned GCS bucket.
- Rollback procedure:
  1. List the revisions: gcloud run revisions list --service aeroassist --region us-central1
  2. Send all the traffic to the previous one: gcloud run services update-traffic aeroassist --region us-central1 --to-revisions=PREVIOUS_REVISION=100

  It is also possible to run cd again from the previous release branch. The details are in docs/RUNBOOK.md.
- Cleanup command:
  Quick: gcloud run services delete aeroassist --region us-central1 --project PROJECT_ID
  Full, only the resources of this project and by name: terraform destroy in infra/terraform/projects/dev, then the bootstrap resources. The exact commands are in docs/RUNBOOK.md.

## Next three improvements

1. Add conversation memory with a LangGraph checkpointer in Firestore, keyed by session_id, with personal data redacted and a TTL (for example 24 hours), so follow-ups like "¿y para mañana?" work. The write gate would still require the booking and the flight in the confirming message.
2. Deploy the MCP server as its own service (streamable HTTP with its own identity) and move Qdrant to a dedicated instance, so the API can scale past one instance.
3. Add a release gate in CD with the paid evaluations (block a release if faithfulness or the write gate regress), and use Model Armor or GCP Sensitive Data Protection for person names and stronger injection detection.
