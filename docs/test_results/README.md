# Test results

Reports written by make eval-rag, eval-mcp and eval-e2e, named suite-label.json. Each holds a
summary, per-case rows and run metadata (target, LLM backend, timestamp).

Only labelled evidence runs are committed (for example baseline-luna or final-remote); the
default label local is git-ignored. Each committed report is listed below with what it shows.

| Report | What it shows |
|---|---|
| classify-6-luna-v1.json, classify-6-sol-v1.json | Classifier v1 prompt (2026-11-15). Luna routed every "Confirmo …" message to injection (false-injection rate 13.5%), because the classifier did not see confirm_action. Run after raising max tokens: at 256, reasoning tokens used the whole budget on 6 cases and no JSON came back |
| classify-6-luna.json, classify-6.1-terra.json, classify-6-sol.json | Classifier v2 prompt (confirm_action passed in; asking about limits is policy, not injection). Route accuracy: Luna 0.946, 6.1-Terra 1.0, Sol 1.0; injection recall 1.0 for all; false injections: Luna 0.058, others 0. p50: Luna 1.6 s, Terra 1.5 s, Sol 1.2 s. Caveat: v2 was written after reading v1 errors on these same 56 labels, so these scores are optimistic |
| mcp-tools-6-luna.json, e2e-tools-6-luna.json | MCP stage (commit 6) with real models: classifier gpt-6.1-terra, tools and answer gpt-6-luna. MCP suite (24 cases): status, tool-call accuracy and F1, required tools, confirmation and evidence all 1.0; 0 write-gate violations; p50 6.1 s, p95 8.9 s. E2E: public 7/8, 23/30 overall; all 7 failures are missing policy citations (RAG arrives in commit 7) |
| mcp-tools-6.1-terra.json, e2e-tools-6.1-terra.json | Same suites with gpt-6.1-terra as the tools model, run before two label fixes (search cases now include the default max_results=3, which lowered F1 to 0.83 for both models). Scores matched Luna exactly, so Luna stays as the tools model at a tenth of the price |

**Tools model choice (D52, 2026-11-15): gpt-6-luna.** It matched gpt-6.1-terra on every MCP and
E2E metric. Most of the MCP latency is sequential model calls (classifier, one or two tool
rounds, answer); the MCP calls themselves take a few milliseconds.

**Classifier choice (D52, 2026-11-15): gpt-6.1-terra.** It ties gpt-6-sol on accuracy (both 1.0
route accuracy, 0 false injections) at half the price, and beats gpt-6-luna, which wrongly blocks
about 6% of legitimate requests with a 400. Price per 1M tokens: input $1.00, output $5.00, cached
input $0.05. At roughly 500 input and up to about 300 output tokens (mostly reasoning) per call,
one classification costs about $0.002. The classifier prompt (about 490 tokens) is below the
1024-token minimum for prompt caching, so caching does not apply to it yet. The answer model is
chosen later, once answers have evidence to work with (commits 6-7).

Write-gate violations must stay at 0 in every report; the runner exits 1 otherwise.
