"""Command-line runner for the RAG, MCP and end-to-end evaluation suites.

Each suite sends its dataset to the chat endpoint, either in-process against the app or
over HTTP against a deployed base URL, and writes a JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import pathlib
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from fastapi import testclient

from app import config, errors, graph, llm
from app import main as app_main
from evaluation import datasets, metrics

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "test_results"
KNOWLEDGE_BASE = pathlib.Path(__file__).resolve().parent.parent / "knowledge_base"


class ChatClient(Protocol):
    """Anything with an httpx-style post method."""

    def post(self, url: str, *, json: Any) -> httpx.Response: ...


def make_client(base_url: str | None) -> ChatClient:
    """Returns an HTTP client for base_url, or an in-process client for the local app.

    Args:
        base_url: Deployed service URL; None evaluates the app in-process.
    """
    if base_url:
        return httpx.Client(base_url=base_url.rstrip("/"), timeout=40)
    settings = config.get_settings().model_copy(
        update={"rate_limit_per_minute": 100_000, "log_level": "WARNING"}
    )
    application = app_main.create_app(settings=settings)
    return testclient.TestClient(application, raise_server_exceptions=False)


def _send(client: ChatClient, payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    response = client.post("/v1/chat", json=payload)
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        body = response.json()
    except ValueError:
        body = {}
    return response.status_code, body, elapsed_ms


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _latency(latencies: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(metrics.percentile(latencies, 50), 1),
        "p95_ms": round(metrics.percentile(latencies, 95), 1),
    }


def _doc_text(doc_id: str) -> str:
    for path in KNOWLEDGE_BASE.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if f"doc_id: {doc_id}\n" in text:
            return text.split("---", 2)[-1].strip()
    return ""


def _judge_scores(cases: list[dict[str, Any]]) -> dict[str, float]:
    """Adds Ragas Faithfulness and AnswerRelevancy using an OpenAI judge (paid)."""
    import asyncio

    import openai
    from ragas import embeddings, llms
    from ragas.metrics import collections as ragas_metrics

    client = openai.AsyncOpenAI()
    llm = llms.llm_factory(os.environ["EVAL_JUDGE_MODEL"], client=client)
    embedder = embeddings.embedding_factory(
        "openai",
        model=os.environ.get("EVAL_EMBEDDING_MODEL", "text-embedding-3-small"),
        client=client,
    )
    faithfulness = ragas_metrics.Faithfulness(llm=llm)
    relevancy = ragas_metrics.AnswerRelevancy(llm=llm, embeddings=embedder)

    async def score_all() -> tuple[list[float], list[float]]:
        faith, relevant = [], []
        for case in cases:
            if case["behavior"] == "abstain" or case["status"] != 200:
                continue
            contexts = [_doc_text(doc_id) for doc_id in case["cited"]]
            if contexts:
                result = await faithfulness.ascore(
                    user_input=case["question"],
                    response=case["answer"],
                    retrieved_contexts=contexts,
                )
                faith.append(float(result.value))
            result = await relevancy.ascore(user_input=case["question"], response=case["answer"])
            relevant.append(float(result.value))
        return faith, relevant

    faith, relevant = asyncio.run(score_all())
    return {"faithfulness": _mean(faith), "answer_relevancy": _mean(relevant)}


def run_rag(client: ChatClient, judge: bool) -> dict[str, Any]:
    """Citation quality, abstention behavior and optional LLM-judged grounding."""
    rows = []
    for case in datasets.load(datasets.DATASETS_DIR / "rag.jsonl", datasets.RagCase):
        status, body, elapsed = _send(client, {"message": case.question})
        cited = [item.get("doc_id") for item in body.get("citations", [])]
        relevant = set(case.expected_doc_ids) | set(case.acceptable_doc_ids)
        abstained = bool(body.get("abstained"))
        if case.behavior == "abstain":
            behavior_ok = abstained and not cited
        else:
            behavior_ok = not abstained and (
                case.behavior == "answer" or body.get("grounding") == "qualified"
            )
        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "behavior": case.behavior,
                "status": status,
                "cited": cited,
                "answer": body.get("answer", ""),
                "precision": metrics.id_precision(cited, relevant),
                "recall": metrics.id_recall(cited, case.expected_doc_ids),
                "irrelevant_citations": sorted(set(cited) - relevant),
                "behavior_ok": behavior_ok,
                "evidence_ok": metrics.contains_any(body, case.must_contain_any),
                "latency_ms": round(elapsed, 1),
            }
        )
    answered = [row for row in rows if row["behavior"] != "abstain"]
    total_citations = sum(len(row["cited"]) for row in rows)
    summary = {
        "cases": len(rows),
        "errors": sum(row["status"] != 200 for row in rows),
        "cited_cases": sum(bool(row["cited"]) for row in answered),
        "citation_precision": _mean([row["precision"] for row in answered if row["cited"]]),
        "citation_recall": _mean([row["recall"] for row in answered]),
        "irrelevant_citation_rate": round(
            sum(len(row["irrelevant_citations"]) for row in rows) / total_citations, 4
        )
        if total_citations
        else 0.0,
        "behavior_accuracy": _mean([float(row["behavior_ok"]) for row in rows]),
        "evidence_rate": _mean([float(row["evidence_ok"]) for row in rows]),
        **_latency([row["latency_ms"] for row in rows]),
    }
    if judge:
        summary.update(_judge_scores(rows))
    return {"summary": summary, "cases": rows}


def run_mcp(client: ChatClient, judge: bool) -> dict[str, Any]:
    """Tool selection, argument accuracy, confirmation behavior and write-gate safety."""
    del judge
    rows = []
    for case in datasets.load(datasets.DATASETS_DIR / "mcp.jsonl", datasets.McpCase):
        payload = case.request.model_dump(exclude_none=True)
        status, body, elapsed = _send(client, payload)
        executed = body.get("tool_calls", [])
        names = {call.get("name") for call in executed}
        row: dict[str, Any] = {
            "id": case.id,
            "status": status,
            "status_ok": status == case.expected_status,
            "executed": [
                {"name": c.get("name"), "arguments": c.get("arguments")} for c in executed
            ],
            "gate_violation": metrics.write_gate_violated(case.request.confirm_action, body),
            "forbidden_called": sorted(names & set(case.forbid_tools)),
            "latency_ms": round(elapsed, 1),
        }
        if status == 200:
            row["confirmation_ok"] = body.get("requires_confirmation") is case.requires_confirmation
            row["evidence_ok"] = metrics.contains_any(body, case.must_contain_any)
            row["required_tools_ok"] = metrics.subset_tools_match(case.expected_tools, executed)
            if case.tool_match == "exact":
                row.update(
                    metrics.ragas_tool_scores(case.request.message, case.expected_tools, executed)
                )
        rows.append(row)
    ok = [row for row in rows if row["status"] == 200]
    exact = [row for row in ok if "tool_call_accuracy" in row]
    summary = {
        "cases": len(rows),
        "status_accuracy": _mean([float(row["status_ok"]) for row in rows]),
        "write_gate_violations": sum(row["gate_violation"] for row in rows),
        "forbidden_tool_calls": sum(bool(row["forbidden_called"]) for row in rows),
        "tool_call_accuracy": _mean([row["tool_call_accuracy"] for row in exact]),
        "tool_call_f1": _mean([row["tool_call_f1"] for row in exact]),
        "required_tools_rate": _mean([float(row["required_tools_ok"]) for row in ok]),
        "confirmation_accuracy": _mean([float(row["confirmation_ok"]) for row in ok]),
        "evidence_rate": _mean([float(row["evidence_ok"]) for row in ok]),
        **_latency([row["latency_ms"] for row in rows]),
    }
    return {"summary": summary, "cases": rows}


def run_e2e(client: ChatClient, judge: bool) -> dict[str, Any]:
    """Public golden cases plus private-style variants, scored black-box."""
    del judge
    cases = datasets.load(datasets.PUBLIC_GOLDEN, datasets.E2ECase) + datasets.load(
        datasets.DATASETS_DIR / "e2e.jsonl", datasets.E2ECase
    )
    rows = []
    for case in cases:
        status, body, elapsed = _send(client, case.request.model_dump(exclude_none=True))
        failures = metrics.check_e2e(case.expect, status, body)
        if metrics.write_gate_violated(case.request.confirm_action, body):
            failures.append("write_gate_violation")
        rows.append(
            {"id": case.id, "status": status, "failures": failures, "latency_ms": round(elapsed, 1)}
        )
    public = [row for row in rows if row["id"].startswith("public-")]
    summary = {
        "cases": len(rows),
        "pass_rate": _mean([float(not row["failures"]) for row in rows]),
        "public_passed": f"{sum(not row['failures'] for row in public)}/{len(public)}",
        "write_gate_violations": sum("write_gate_violation" in row["failures"] for row in rows),
        "server_errors": sum(row["status"] >= 500 for row in rows),
        **_latency([row["latency_ms"] for row in rows]),
    }
    return {"summary": summary, "cases": rows}


def run_classify(client: ChatClient, judge: bool) -> dict[str, Any]:
    """Classifier route and flag accuracy, calling the configured classifier model directly."""
    del client, judge
    settings = config.get_settings()
    provider = llm.build_llm(settings)
    fields = ("needs_tools", "needs_policy", "write_requested")

    async def classify_all() -> list[dict[str, Any]]:
        rows = []
        for label in datasets.classifier_labels():
            started = time.perf_counter()
            try:
                got = await graph.classify(provider, label.message, label.confirm_action)
            except errors.AppError as exc:
                rows.append(
                    {
                        "id": label.id,
                        "expected_route": label.route,
                        "route": None,
                        "route_ok": False,
                        "error": f"{exc.code}: {type(exc.__cause__).__name__}",
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                )
                continue
            row: dict[str, Any] = {
                "id": label.id,
                "expected_route": label.route,
                "route": got.route,
                "route_ok": got.route == label.route,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            if label.route == "answer" and got.route == "answer":
                for field in fields:
                    if getattr(label, field) is not None:
                        row[f"{field}_ok"] = getattr(got, field) == getattr(label, field)
            rows.append(row)
        return rows

    rows = asyncio.run(classify_all())
    summary: dict[str, Any] = {
        "cases": len(rows),
        "model": settings.classifier_model if settings.llm_backend != "fake" else "fake",
        "errors": sum("error" in row for row in rows),
        "route_accuracy": _mean([float(row["route_ok"]) for row in rows]),
        "injection_recall": _mean(
            [float(row["route_ok"]) for row in rows if row["expected_route"] == "injection"]
        ),
        "false_injection_rate": _mean(
            [
                float(row["route"] == "injection")
                for row in rows
                if row["expected_route"] != "injection"
            ]
        ),
    }
    for field in fields:
        summary[f"{field}_accuracy"] = _mean(
            [float(row[f"{field}_ok"]) for row in rows if f"{field}_ok" in row]
        )
    summary.update(_latency([row["latency_ms"] for row in rows]))
    return {"summary": summary, "cases": rows}


SUITES: dict[str, Callable[[ChatClient, bool], dict[str, Any]]] = {
    "classify": run_classify,
    "rag": run_rag,
    "mcp": run_mcp,
    "e2e": run_e2e,
}


def main(argv: list[str] | None = None) -> int:
    """Runs one suite and writes docs/test_results/<suite>-<label>.json.

    Args:
        argv: Command-line arguments; defaults to sys.argv.

    Returns:
        Process exit code: 1 if any write-gate violation occurred, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=sorted(SUITES))
    parser.add_argument("--base-url", default=None, help="Deployed URL; omit for in-process.")
    parser.add_argument("--label", default="local", help="Report name suffix.")
    parser.add_argument("--judge", action="store_true", help="Add paid LLM-judged metrics.")
    args = parser.parse_args(argv)

    report = SUITES[args.suite](make_client(args.base_url), args.judge)
    report["meta"] = {
        "suite": args.suite,
        "label": args.label,
        "target": args.base_url or "in-process",
        "llm_backend": os.environ.get("LLM_BACKEND", "fake"),
        "run_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.suite}-{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report: {out.relative_to(RESULTS_DIR.parent.parent)}")
    return 1 if report["summary"].get("write_gate_violations") else 0


if __name__ == "__main__":
    sys.exit(main())
