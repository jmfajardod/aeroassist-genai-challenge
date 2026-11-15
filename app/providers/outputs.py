"""Typed outputs requested from the LLM.

Every pydantic field is required so the schemas work with strict JSON-schema decoding.
"""

from __future__ import annotations

from typing import Any, Literal

import pydantic

Role = Literal["classifier", "answer", "tools"]


class Classification(pydantic.BaseModel):
    """Routing decision for one message."""

    route: Literal["injection", "help", "unsupported", "answer"]
    needs_tools: bool
    needs_policy: bool
    write_requested: bool


class Answer(pydantic.BaseModel):
    """Grounded answer with the policy chunks it used."""

    answer: str
    cited_chunk_ids: list[str]
    grounding: Literal["grounded", "qualified", "ungrounded"]


class ToolRequest(pydantic.BaseModel):
    """A tool call proposed by the model; arguments are checked per tool before anything runs."""

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class ToolStep(pydantic.BaseModel):
    """The model's next move: tool calls to make, or a final text when it makes none."""

    model_config = pydantic.ConfigDict(frozen=True)

    requests: list[ToolRequest]
    text: str = ""


ToolRound = list[tuple[ToolRequest, dict[str, Any]]]
