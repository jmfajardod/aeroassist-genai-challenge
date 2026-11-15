"""Metric functions shared by the evaluation suites."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

from evaluation import datasets

WRITE_TOOL = "create_reaccommodation_hold"


def id_precision(retrieved: Iterable[str], relevant: Iterable[str]) -> float:
    """Share of retrieved IDs that are relevant; 1.0 when nothing was retrieved."""
    retrieved_set = set(retrieved)
    if not retrieved_set:
        return 1.0
    return len(retrieved_set & set(relevant)) / len(retrieved_set)


def id_recall(retrieved: Iterable[str], reference: Iterable[str]) -> float:
    """Share of reference IDs that were retrieved; 1.0 when there is no reference."""
    reference_set = set(reference)
    if not reference_set:
        return 1.0
    return len(set(retrieved) & reference_set) / len(reference_set)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile.

    Args:
        values: Observations.
        q: Percentile in [0, 100].

    Returns:
        The percentile, or 0.0 for no observations.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return float(ordered[rank - 1])


def answer_text(body: dict[str, Any]) -> str:
    """Normalizes the answer exactly like the public evaluator does."""
    return json.dumps(body.get("answer", ""), ensure_ascii=False).lower()


def contains_any(body: dict[str, Any], tokens: Sequence[str]) -> bool:
    """True if no tokens are given or any token appears in the answer."""
    text = answer_text(body)
    return not tokens or any(token.lower() in text for token in tokens)


def write_gate_violated(confirm_action: bool, body: dict[str, Any]) -> bool:
    """True if a write ran without confirmation or more than one write ran."""
    writes = [call for call in body.get("tool_calls", []) if call.get("name") == WRITE_TOOL]
    return len(writes) > 1 or (bool(writes) and not confirm_action)


def check_e2e(expect: datasets.E2EExpect, status: int, body: dict[str, Any]) -> list[str]:
    """Applies public-golden semantics plus the extra E2E expectations.

    Args:
        expect: Expectations for the case.
        status: HTTP status received.
        body: Parsed JSON body.

    Returns:
        Failure labels; empty when the case passes.
    """
    if status != expect.status:
        return [f"status={status}"]
    if status != 200:
        code = body.get("error", {}).get("code")
        return [f"error_code={code}"] if expect.error_code and code != expect.error_code else []
    failures = []
    tools = {call.get("name") for call in body.get("tool_calls", [])}
    citations = {item.get("doc_id") for item in body.get("citations", [])}
    if expect.tool_any and not tools & set(expect.tool_any):
        failures.append("missing_tool")
    if tools & set(expect.forbid_tools) or (expect.no_tools and tools):
        failures.append("forbidden_tool")
    if expect.citation_any and not citations & set(expect.citation_any):
        failures.append("missing_citation")
    if expect.no_citations and citations:
        failures.append("unexpected_citation")
    if body.get("requires_confirmation") is not expect.requires_confirmation:
        failures.append("requires_confirmation")
    if not contains_any(body, expect.must_contain_any):
        failures.append("answer_evidence")
    text = answer_text(body)
    if any(token.lower() not in text for token in expect.must_contain_all):
        failures.append("answer_evidence_all")
    return failures


def arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """True if every expected argument equals the executed one."""
    return all(actual.get(key) == value for key, value in expected.items())


def subset_tools_match(
    expected: Sequence[datasets.ExpectedToolCall], executed: Sequence[dict[str, Any]]
) -> bool:
    """True if each expected call has an executed call with the same name and arguments."""
    return all(
        any(
            call.get("name") == want.name
            and arguments_match(want.arguments, call.get("arguments", {}))
            for call in executed
        )
        for want in expected
    )


def ragas_tool_scores(
    message: str,
    expected: Sequence[datasets.ExpectedToolCall],
    executed: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Scores executed tool calls against the reference with Ragas, without an LLM.

    Ragas is imported here because it lives in the optional eval dependency group.

    Args:
        message: User message.
        expected: Reference tool calls.
        executed: Tool call traces from the response.

    Returns:
        tool_call_accuracy and tool_call_f1 in [0, 1].
    """
    if not expected and not executed:
        return {"tool_call_accuracy": 1.0, "tool_call_f1": 1.0}
    from ragas import messages as ragas_messages
    from ragas.metrics import collections as ragas_metrics

    calls = [
        ragas_messages.ToolCall(name=call.get("name", ""), args=call.get("arguments", {}))
        for call in executed
    ]
    reference = [ragas_messages.ToolCall(name=c.name, args=c.arguments) for c in expected]
    user_input = [
        ragas_messages.HumanMessage(content=message),
        ragas_messages.AIMessage(content="", tool_calls=calls or None),
    ]

    async def score() -> dict[str, float]:
        accuracy = await ragas_metrics.ToolCallAccuracy(strict_order=False).ascore(
            user_input=user_input, reference_tool_calls=reference
        )
        f1 = await ragas_metrics.ToolCallF1().ascore(
            user_input=user_input, reference_tool_calls=reference
        )
        return {"tool_call_accuracy": float(accuracy.value), "tool_call_f1": float(f1.value)}

    return asyncio.run(score())
