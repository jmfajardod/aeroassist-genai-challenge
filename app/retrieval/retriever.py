"""Builds the Qdrant-backed retriever from settings and applies the minimum-score filter."""

from __future__ import annotations

import asyncio

import qdrant_client

from app import rag
from app.core import config
from app.retrieval import chunker, embeddings, repository


def build_embedder(settings: config.Settings) -> embeddings.Embedder:
    """Returns the configured embedder."""
    if settings.embedding_backend == "fake":
        return embeddings.HashingEmbedder()
    return embeddings.FastEmbedEmbedder(
        settings.dense_embedding_model,
        settings.sparse_embedding_model,
        settings.embedding_cache_dir,
    )


def build_repository(settings: config.Settings) -> repository.QdrantRepository:
    """Connects to Qdrant: the server at QDRANT_URL, or an in-process index when it is unset.

    Args:
        settings: Application settings.

    Returns:
        A repository whose collections exist.
    """
    if settings.qdrant_url:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        client = qdrant_client.QdrantClient(
            url=settings.qdrant_url, api_key=api_key, timeout=settings.qdrant_timeout_seconds
        )
    else:
        client = qdrant_client.QdrantClient(location=":memory:")
    repo = repository.QdrantRepository(
        client, build_embedder(settings), settings.qdrant_collection_prefix
    )
    repo.ensure_collections()
    return repo


class QdrantRetriever:
    """Retriever over the repository with two thresholds on the [0, 1] score.

    The evidence gate decides abstention: if even the best chunk scores below it, nothing is
    returned. The per-chunk floor is lower and only drops noise, so chunks found by BM25 with a
    modest dense score survive; precision then comes from the model citing only what it used.

    Args:
        repo: Qdrant repository.
        min_score: Per-chunk floor.
        evidence_score: Minimum best score for the request to have any evidence.
    """

    def __init__(
        self, repo: repository.QdrantRepository, min_score: float, evidence_score: float
    ) -> None:
        self._repo = repo
        self._min_score = min_score
        self._evidence_score = evidence_score
        self._version = repo.index_version()

    @property
    def index_version(self) -> str | None:
        """Version of the indexed corpus when the retriever was built."""
        return self._version

    async def search(self, query: str, *, top_k: int) -> list[rag.RetrievedChunk]:
        """Runs the blocking hybrid search in a worker thread and filters by score.

        Args:
            query: User question.
            top_k: Maximum number of chunks.

        Returns:
            Chunks that pass both thresholds, in fused rank order.
        """
        return self.apply_thresholds(await asyncio.to_thread(self._repo.search, query, top_k))

    def apply_thresholds(self, chunks: list[rag.RetrievedChunk]) -> list[rag.RetrievedChunk]:
        """Applies the evidence gate, then the per-chunk floor.

        Args:
            chunks: Search results.

        Returns:
            An empty list when the best score is below the gate, else chunks above the floor.
        """
        if max((chunk.score for chunk in chunks), default=0.0) < self._evidence_score:
            return []
        return [chunk for chunk in chunks if chunk.score >= self._min_score]


def expected_index_version(settings: config.Settings) -> str:
    """Index version of the knowledge base shipped with this build."""
    documents = chunker.load_corpus(settings.knowledge_base_path)
    return chunker.index_version({d.doc_id: d.content_hash for d in documents})
