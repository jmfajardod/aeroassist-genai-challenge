"""Stable HTTP schemas used by public and private evaluators."""

from __future__ import annotations

import datetime
from typing import Any, Literal

import pydantic


class ChatRequest(pydantic.BaseModel):
    """Chat input; message length limits apply after trimming."""

    model_config = pydantic.ConfigDict(extra="forbid")

    message: str = pydantic.Field(min_length=1, max_length=4_000)
    session_id: str | None = pydantic.Field(default=None, max_length=128)
    confirm_action: bool = False

    @pydantic.field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, value: object) -> object:
        """Trims surrounding whitespace before the length checks run."""
        return value.strip() if isinstance(value, str) else value


class Citation(pydantic.BaseModel):
    """A knowledge-base chunk that supports the answer."""

    doc_id: str
    title: str
    chunk_id: str
    score: float = pydantic.Field(ge=0.0, le=1.0)


class ToolCallTrace(pydantic.BaseModel):
    """An executed tool call, with validated arguments only."""

    name: str
    arguments: dict[str, Any]
    status: str
    duration_ms: int = pydantic.Field(ge=0)


class Similarity(pydantic.BaseModel):
    """Retrieval score summary on the [0, 1] scale."""

    top: float = pydantic.Field(ge=0.0, le=1.0)
    mean: float = pydantic.Field(ge=0.0, le=1.0)


class Usage(pydantic.BaseModel):
    """LLM token usage and estimated cost of one request."""

    llm_calls: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_usd: float | None


class ChatResponse(pydantic.BaseModel):
    """Chat output: contract fields first, then diagnostic extras."""

    answer: str
    citations: list[Citation] = pydantic.Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = pydantic.Field(default_factory=list)
    requires_confirmation: bool = False
    request_id: str
    model: str
    latency_ms: int = pydantic.Field(ge=0)
    abstained: bool = False
    grounding: Literal["grounded", "qualified", "ungrounded", "not_applicable"] = "not_applicable"
    similarity: Similarity | None = None
    route: str | None = None
    index_version: str | None = None
    latency_breakdown_ms: dict[str, int] = pydantic.Field(default_factory=dict)
    usage: Usage | None = None


class HealthResponse(pydantic.BaseModel):
    """Liveness body."""

    status: Literal["ok"] = "ok"
    version: str


class ReadyResponse(pydantic.BaseModel):
    """Readiness body, returned with 200 when ready and 503 otherwise."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


class ErrorDetail(pydantic.BaseModel):
    """Machine-readable code and client-safe message."""

    code: str
    message: str


class ErrorResponse(pydantic.BaseModel):
    """Error envelope shared by every endpoint."""

    error: ErrorDetail
    request_id: str | None


class CapabilityLevels(pydantic.BaseModel):
    """Per-level flags; true only when backed by passing evidence."""

    foundation: bool
    core: bool
    tools_mcp: bool
    rag: bool
    production: bool


class CapabilitiesResponse(pydantic.BaseModel):
    """Capabilities report for evaluators."""

    challenge_version: str
    levels: CapabilityLevels
    llm_backend: str
    tool_transport: str
    variant_id: str | None = None
    reference_date: datetime.date | None = None
