# Test results

Reports written by make eval-rag, eval-mcp and eval-e2e, named suite-label.json. Each holds a
summary, per-case rows and run metadata (target, LLM backend, timestamp).

Only labelled evidence runs are committed (for example baseline-luna or final-remote); the
default label local is git-ignored. Each committed report is listed below with what it shows.

| Report | What it shows |
|---|---|

Write-gate violations must stay at 0 in every report; the runner exits 1 otherwise.
