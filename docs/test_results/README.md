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
| retrieval-emb-{minilm,mpnet,jina-es,potion,e5-large}.json | Retrieval only, no LLM (make eval-retrieval): five FastEmbed dense models, each hybrid with Spanish BM25, top 3, recall@3 1.0 for all. hit@1 / MRR: MiniLM 0.90/0.95, mpnet 0.90/0.95, Jina-es 1.00/1.00, potion 0.95/0.98, e5-large 1.00/1.00. Score gap between the weakest answerable and the strongest out-of-corpus question: MiniLM 0.08, mpnet 0.09, Jina-es 0.14, potion overlapping, e5-large 0.03. Query p50: 26, 73, 70, 1, 139 ms. Model size: 0.22, 1.0, 0.64, 0.5, 2.24 GB. Evidence-gate rows in each file use the MiniLM-calibrated gate (0.45), so the Jina row under-reports; its own gate would be about 0.35 |
| rag-rag-v1.json | RAG suite, retrieval only when the classifier set needs_policy. Citation precision 1.0, recall 0.95, 0 irrelevant citations; rag-02 got no citation because the classifier missed the policy flag |
| rag-rag-v2.json, e2e-rag-v2.json, mcp-rag-v2.json | Final commit-7 stack (retrieval on every answer route, MiniLM + BM25, evidence gate 0.45, floor 0.20; real LLMs). RAG: citation precision 1.0, recall 1.0, 0 irrelevant citations, behavior 0.92 (rag-02 and rag-14 answered as grounded where the labels say qualify; flagged for the spot-check). E2E: 30/30, public 8/8. MCP: 1.0 on every metric, 0 write-gate violations |

**Embedding choice (D49, D95): paraphrase-multilingual-MiniLM-L12-v2 + Qdrant/bm25 (Spanish).**
Published MTEB numbers rank multilingual-e5-large highest and potion lower on retrieval, and
Jina reports v2-base-es beating multilingual E5 on Spanish retrieval; MiniLM has no published
retrieval score, so the choice rests on our measurements. Jina-es is the best quality on our
data (perfect ranking, widest gap) but is 3x larger, 2.7x slower and needs about 270 MB more RAM,
which moves Cloud Run from 1 GiB to 2 GiB. e5-large ranks perfectly but its scores are compressed
(gap 0.03, too thin for a stable gate) and it is 2.24 GB. MiniLM already gives citation precision
and recall 1.0 end to end because the relevant chunk is always in the top 3, so it is the best
balance; Jina-es (gate about 0.35) is the quality upgrade, a configuration change away. The
thresholds were calibrated on the same 24 RAG cases, so the margins are provisional.
| rag-answer-{luna,terra}-{low,none}.json | Answer model and reasoning comparison with the Ragas judge (gpt-6.1-terra): faithfulness luna/low 0.85, luna/none 0.82, terra/low 0.90, terra/none 0.94; citation precision 1.0 for all. gpt-6-luna rejects reasoning effort `minimal` |
| classify-6.1-terra-none.json | Classifier at reasoning `none`: same scores as `low` on the 56 labels; later a borderline question was flagged as injection in one run, so the classifier went back to `low` |
| mcp-tools-luna-none.json | Tools model gpt-6-luna at `none`: 1.0 on every MCP metric, same as `low` |
| rag-faith-p1..p4.json, rag-faith-p2b.json | Answer-prompt iterations for faithfulness: p1 original 0.939; p2 evidence-only wording 0.965 and 0.954 (p2b); p3 missing-data rule 0.904 (asks for data the policy does not mention, plus a false 400); p4 asks only when the policy cannot answer and no bare yes/no, 0.958 with citation recall 1.0 |
| rag-final.json, e2e-final.json, mcp-final.json | **Final code** (prompt p4; classifier gpt-6.1-terra/low, tools gpt-6-luna/none, answers gpt-6.1-terra/none; MiniLM + BM25). RAG: faithfulness 0.947 (0.958 in p4 on the same code, mean about 0.95), citation precision 1.0, recall 0.95, 0 irrelevant citations, 0 errors. E2E 30/30, public 8/8. MCP 1.0 on every metric, 0 write-gate violations. Remaining faithfulness losses are close paraphrases the judge reads strictly (rag-02, rag-10); behavior misses are the qualify labels flagged in the spot-check log |
| e2e-ops.json, mcp-ops.json, rag-ops.json | Commit 8 (tracing, PII redaction on LLM input, usage/cost accounting), real models, requests traced to Langfuse. E2E 30/30 (public 8/8), p50 3.9 s / p95 8.0 s, **$0.0013 per request**. MCP 1.0 on every metric including evidence (after the not-found rule), **$0.0016 per request**. RAG citation precision and recall 1.0, 0 irrelevant citations, **$0.0019 per request**. The not-found prompt rule was added after the E2E and RAG runs; it only applies to tool results with found=false, which those suites do not contain |

**Answer prompt and models (D84, D101, D105).** The answer prompt requires every statement to be
backed by the evidence and close to its wording, forbids added advice, and asks for operational
data only when the policy cannot answer. Faithfulness went from 0.94 to about 0.95; the judge
is the same model family as the answer model, so treat absolute values with that caveat.

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
