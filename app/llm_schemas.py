"""Structured outputs requested from the LLM.

Every field is required so the schemas work with strict JSON-schema decoding.
"""

from __future__ import annotations

from typing import Literal

import pydantic


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
