"""Retrieval boundary used by the answer graph."""

from __future__ import annotations

from typing import Protocol

import pydantic


class RetrievedChunk(pydantic.BaseModel):
    """A knowledge-base chunk with a relevance score in [0, 1]."""

    model_config = pydantic.ConfigDict(frozen=True)

    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: float = pydantic.Field(ge=0.0, le=1.0)


class Retriever(Protocol):
    """Anything that returns the most relevant chunks for a query."""

    @property
    def index_version(self) -> str | None:
        """Version of the indexed corpus, shown in responses and readiness."""

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Returns the best-matching chunks.

        Args:
            query: User question.
            top_k: Maximum number of chunks.

        Returns:
            Chunks ordered by descending score.
        """
