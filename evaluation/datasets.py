"""Typed evaluation dataset records and loaders."""

from __future__ import annotations

import pathlib
from typing import Any, Literal, TypeVar

import pydantic

DATASETS_DIR = pathlib.Path(__file__).resolve().parent / "datasets"
PUBLIC_GOLDEN = pathlib.Path(__file__).resolve().parent / "public_golden.jsonl"

TOOL_NAMES = frozenset(
    {
        "get_flight_status",
        "get_booking_context",
        "search_reaccommodation",
        "create_reaccommodation_hold",
    }
)


class _Record(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class ChatRequestPayload(_Record):
    """Request body sent to the chat endpoint."""

    message: str
    session_id: str | None = None
    confirm_action: bool = False


class RagCase(_Record):
    """Policy question labelled with the documents that must support the answer.

    Attributes:
        expected_doc_ids: Documents required for a complete answer; empty for abstentions.
        acceptable_doc_ids: Related documents that are relevant but not required.
        behavior: Whether the assistant should answer, answer with caveats, or abstain.
    """

    id: str
    question: str
    behavior: Literal["answer", "qualify", "abstain"]
    expected_doc_ids: list[str]
    acceptable_doc_ids: list[str] = []
    reference_answer: str
    must_contain_any: list[str] = []


class ExpectedToolCall(_Record):
    """A tool call the assistant must make; arguments are compared key by key."""

    name: str
    arguments: dict[str, Any]


class McpCase(_Record):
    """Operational request labelled with the expected tool calls.

    Attributes:
        tool_match: exact scores the executed calls against expected_tools with Ragas;
            subset only requires expected_tools to be present and allows extra read tools.
    """

    id: str
    request: ChatRequestPayload
    expected_status: int = 200
    expected_tools: list[ExpectedToolCall]
    tool_match: Literal["exact", "subset"] = "exact"
    requires_confirmation: bool = False
    forbid_tools: list[str] = []
    must_contain_any: list[str] = []
    notes: str = ""


class E2EExpect(_Record):
    """Black-box expectations, a superset of the public golden format."""

    status: int = 200
    error_code: str | None = None
    tool_any: list[str] = []
    forbid_tools: list[str] = []
    no_tools: bool = False
    citation_any: list[str] = []
    no_citations: bool = False
    requires_confirmation: bool = False
    must_contain_any: list[str] = []
    must_contain_all: list[str] = []


class E2ECase(_Record):
    """End-to-end case in the public golden format."""

    id: str
    request: ChatRequestPayload
    expect: E2EExpect


_T = TypeVar("_T", bound=pydantic.BaseModel)


def load(path: pathlib.Path, model: type[_T]) -> list[_T]:
    """Parses a JSONL dataset.

    Args:
        path: JSONL file with one record per line.
        model: Record type to validate each line against.

    Returns:
        The validated records in file order.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]
