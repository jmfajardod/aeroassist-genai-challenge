"""Integrity of the evaluation datasets and the safety-critical metric logic."""

from __future__ import annotations

import datetime
import pathlib
import re

import pytest

from app.core import constants
from evaluation import datasets, metrics

_KB_DOC_IDS = {
    match.group(1)
    for path in pathlib.Path("knowledge_base").glob("*.md")
    if (match := re.search(r"^doc_id: (\S+)$", path.read_text(encoding="utf-8"), re.MULTILINE))
}
_FORMATS = {
    "flight_number": re.compile(r"^[A-Z]{2}\d{1,4}$"),
    "booking_code": re.compile(r"^[A-Z0-9]{6}$"),
    "origin": re.compile(r"^[A-Z]{3}$"),
    "destination": re.compile(r"^[A-Z]{3}$"),
}

RAG = datasets.load(datasets.DATASETS_DIR / "rag.jsonl", datasets.RagCase)
MCP = datasets.load(datasets.DATASETS_DIR / "mcp.jsonl", datasets.McpCase)
E2E = datasets.load(datasets.DATASETS_DIR / "e2e.jsonl", datasets.E2ECase)
PUBLIC = datasets.load(datasets.PUBLIC_GOLDEN, datasets.E2ECase)


def test_case_ids_are_unique_across_datasets() -> None:
    ids = [case.id for case in [*RAG, *MCP, *E2E, *PUBLIC]]
    assert len(ids) == len(set(ids))


def test_rag_labels_reference_existing_documents() -> None:
    for case in RAG:
        assert set(case.expected_doc_ids) | set(case.acceptable_doc_ids) <= _KB_DOC_IDS, case.id
        assert (case.behavior == "abstain") == (not case.expected_doc_ids), case.id


def test_e2e_citations_and_tools_are_valid() -> None:
    for case in E2E:
        assert set(case.expect.citation_any) <= _KB_DOC_IDS, case.id
        assert set(case.expect.tool_any) | set(case.expect.forbid_tools) <= constants.TOOL_NAMES


def test_mcp_expected_arguments_are_well_formed() -> None:
    for case in MCP:
        for call in case.expected_tools:
            assert call.name in constants.TOOL_NAMES, case.id
            for key, value in call.arguments.items():
                if key == "departure_date":
                    datetime.date.fromisoformat(value)
                elif key in _FORMATS:
                    assert _FORMATS[key].match(value), (case.id, key, value)


def test_no_dataset_expects_a_write_without_confirmation() -> None:
    for case in MCP:
        if any(call.name == constants.WRITE_TOOL for call in case.expected_tools):
            assert case.request.confirm_action, case.id
    for case in [*E2E, *PUBLIC]:
        if constants.WRITE_TOOL in case.expect.tool_any:
            assert case.request.confirm_action, case.id


@pytest.mark.parametrize(
    ("confirm", "names", "violated"),
    [
        (False, ["create_reaccommodation_hold"], True),
        (True, ["create_reaccommodation_hold", "create_reaccommodation_hold"], True),
        (True, ["get_booking_context", "create_reaccommodation_hold"], False),
        (False, ["get_booking_context"], False),
    ],
)
def test_write_gate_violation_detection(confirm: bool, names: list[str], violated: bool) -> None:
    body = {"tool_calls": [{"name": name, "arguments": {}} for name in names]}
    assert metrics.write_gate_violated(confirm, body) is violated


def test_e2e_check_matches_public_semantics() -> None:
    expect = datasets.E2EExpect(
        tool_any=["get_flight_status"], requires_confirmation=False, must_contain_any=["cancel"]
    )
    good = {
        "answer": "El vuelo está CANCELADO",
        "tool_calls": [{"name": "get_flight_status"}],
        "requires_confirmation": False,
    }
    assert metrics.check_e2e(expect, 200, good) == []
    assert metrics.check_e2e(expect, 200, {**good, "requires_confirmation": None}) == [
        "requires_confirmation"
    ]
    assert metrics.check_e2e(expect, 502, good) == ["status=502"]
