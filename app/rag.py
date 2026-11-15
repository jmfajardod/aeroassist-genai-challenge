"""Retrieval boundary used by the answer graph."""

from __future__ import annotations

import dataclasses
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class RetrievedChunk:
    """A knowledge-base chunk with a relevance score in [0, 1]."""

    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: float


class Retriever(Protocol):
    """Anything that returns the most relevant chunks for a query."""

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...
