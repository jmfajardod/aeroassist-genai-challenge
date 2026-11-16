"""Mirrors the evaluation datasets into Langfuse Datasets for experiments; idempotent.

Usage: python -m evaluation.langfuse_sync
"""

from __future__ import annotations

import json
import sys

from app.core import config
from app.observability import tracing
from evaluation import datasets


def main() -> int:
    """Upserts every RAG, MCP and E2E case as a Langfuse dataset item keyed by its case ID.

    Returns:
        Process exit code; 1 when Langfuse is not configured.
    """
    client = tracing.Tracer(config.get_settings()).client
    if client is None:
        print("Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY).")
        return 1
    suites = {
        "rag": datasets.load(datasets.DATASETS_DIR / "rag.jsonl", datasets.RagCase),
        "mcp": datasets.load(datasets.DATASETS_DIR / "mcp.jsonl", datasets.McpCase),
        "e2e": datasets.load(datasets.DATASETS_DIR / "e2e.jsonl", datasets.E2ECase),
    }
    report = {}
    for suite, cases in suites.items():
        name = f"aeroassist-{suite}"
        client.create_dataset(name=name, description=f"AeroAssist {suite} evaluation cases")
        for case in cases:
            record = case.model_dump(mode="json")
            client.create_dataset_item(
                dataset_name=name,
                id=f"{suite}-{case.id}",
                input=record.get("request") or {"message": record.get("question")},
                expected_output={
                    k: v for k, v in record.items() if k not in ("request", "question")
                },
            )
        report[name] = len(cases)
    client.flush()
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
