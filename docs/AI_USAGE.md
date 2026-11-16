# AI assistance disclosure

Using AI is allowed. This disclosure is not scored by the amount of AI used; it supports a fair discussion of ownership.

## Tools used

| Tool/model | Tasks | Approximate share | How output was verified |
| --- | --- | ---: | --- |
| Claude Code (Opus 5.5) | Generation of the evaluation datasets, and code generation for the application, tests, infrastructure and workflows | 90% | 86 tests and lint in CI, my spot-check of the datasets, the official public evaluator (8/8 local, in Docker and on Cloud Run), evaluation runs with real models (docs/test_results/), and my review of every decision |

## Material changes made after AI output

These are the ten most important changes I made or asked for after reviewing the AI output:

1. Classifier prompt: the first evaluation with real models showed false prompt injections on "Confirmo …" messages (13.5% with gpt-6-luna), so the classifier now receives confirm_action and asking about the assistant limits is not treated as an attack.
2. Write gate: the first version returned 400 for a booking that was not found, which contradicted my decision (200 with an explanation and the valid options), so it was corrected.
3. Retrieval thresholds: one threshold on the dense score dropped a relevant chunk found by BM25, so it became an evidence gate on the best chunk plus a low floor per chunk.
4. Embeddings: I asked to compare at least five FastEmbed models and to check them against MTEB. An earlier claim that MiniLM had "the widest score gap" was wrong; MiniLM stays for size and latency, and jina-v2-base-es is documented as the upgrade.
5. Models and reasoning: I asked to benchmark the answer model and the reasoning effort with the Ragas judge, which moved answers to gpt-6.1-terra with reasoning none and showed that GPT-6 rejects the minimal reasoning effort.
6. Faithfulness: I set the target at 0.90 or higher, close to 0.95. The answer prompt went through four versions (0.939, then 0.965 and 0.954, then 0.904, then about 0.95), and one of the AI's rules made it worse and was replaced.
7. Trace masking: reading a real trace from the Langfuse API showed a leak through LangGraph Send objects (synthetic data), so the mask now fails closed.
8. PII: I chose LangChain detectors plus regex first, and left GCP Sensitive Data Protection and Model Armor as future options, instead of a GCP-first design.
9. Code structure: I asked for the Google style guide with module imports and minimal comments, pydantic models instead of dataclasses, one constants module, and a package layout that keeps the starter files where they were.
10. IAM and secrets: I asked for a full audit before the first CD run; it found no secret in git history, logs, state or the image, and it removed two broad roles from the Terraform account that allowed privilege escalation.

## One suggestion rejected and why

These are suggestions from Claude that I rejected:

1. Fake-backend baseline: Claude committed evaluation reports produced by the fake backend as a "baseline". I rejected them, since they measured a keyword stub and not the system, and they could mislead the reviewers; now only runs with real models are committed.
2. Classifier model: Claude recommended gpt-6-sol. I chose gpt-6.1-terra, since it had the same accuracy at half the price.
3. Private repository: Claude recommended a private repository, following the README. I chose a public one, since the submission email asks for a public GitHub URL, and asked for protections so only I can trigger deployments.
4. Actions pinned to commit SHAs: Claude pinned the GitHub Actions to commit SHAs. I kept version tags (v2, v3, v4) for readability and easier updates.
5. Deploying from main: the first workflows ran terraform-apply from main. I asked that terraform-apply and cd run only from release branches, never from main.
6. Skipping the observability phase: Claude suggested moving Langfuse and the usage metrics to the end, or cutting them, to prioritize the deployment. I kept the original plan.
7. Chunking without overlap: Claude kept a hand-written section splitter without overlap. I asked for the LangChain splitters with overlap, so longer documents are handled.
8. Firestore for the document registry: the first plan used Firestore for documents. I limited Firestore to chat session state, and only if it is needed; the document registry stays in Qdrant.
9. Full decision log in DECISIONS.md: Claude kept a long numbered log inside DECISIONS.md. I asked for only the original sections; the log moved to docs/DECISION_LOG.md.
10. Continuing a stale evaluation run: a final evaluation run kept going after a code change. I stopped it, so the final evidence only comes from the final code.

## Files or areas written without AI assistance

- docs/DECISIONS.md: written mainly by me; Claude only rewrote it for better clarity.

## Known uncertainty

- The classifier prompt v2 was written after reading the v1 errors on the same 56 labeled messages, so its accuracy is optimistic.
- The rules of the fake backend were written knowing the datasets, so fake-backend scores are not evidence of quality.
- The Ragas judge (gpt-6.1-terra) is the same model family as the answer model, so it may favor those answers, and each configuration ran once on 24 cases.
- The retrieval thresholds (evidence gate 0.45 and floor 0.20) were calibrated on the same 24 cases they are evaluated on, with margins of about 0.04.
- Two RAG cases (rag-02 and rag-14) are labeled qualify, but the model answers them as grounded with the right citation, which lowers the behavior accuracy.
- An unknown flight and a flight on another route both get a 200 explanation, since the read tools cannot tell them apart.
- Person names are not redacted from logs and traces.
