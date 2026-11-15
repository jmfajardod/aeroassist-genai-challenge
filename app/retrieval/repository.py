"""Qdrant repository with document-level CRUD and hybrid (dense + BM25) search.

Two collections: a vectorless document registry (metadata and content hash) and the chunks.
Updating a document deletes its chunks and inserts the new ones; an unchanged hash is skipped.
"""

from __future__ import annotations

import math
import uuid

import pydantic
import qdrant_client
from qdrant_client import models

from app import rag
from app.retrieval import chunker, embeddings

_DENSE = "dense"
_SPARSE = "bm25"


class DocumentRecord(pydantic.BaseModel):
    """Registry entry for one indexed document."""

    model_config = pydantic.ConfigDict(frozen=True)

    doc_id: str
    title: str
    version: str
    updated: str
    content_hash: str
    chunk_count: int


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def cosine_score(left: list[float], right: list[float]) -> float:
    """Cosine similarity clipped to [0, 1]; this is the score shown to clients.

    Fused RRF scores only rank results; their scale says nothing about relevance.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return max(0.0, min(1.0, dot / norm)) if norm else 0.0


class QdrantRepository:
    """Document CRUD and hybrid search over a Qdrant client (server or local mode).

    Args:
        client: Qdrant client.
        embedder: Dense and sparse embedder.
        prefix: Collection name prefix.
    """

    def __init__(
        self, client: qdrant_client.QdrantClient, embedder: embeddings.Embedder, prefix: str
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._documents = f"{prefix}_documents"
        self._chunks = f"{prefix}_chunks"

    def ensure_collections(self) -> None:
        """Creates both collections and the doc_id payload index if they do not exist."""
        client = self._client
        if not client.collection_exists(self._documents):
            client.create_collection(self._documents, vectors_config={})
        if not client.collection_exists(self._chunks):
            client.create_collection(
                self._chunks,
                vectors_config={
                    _DENSE: models.VectorParams(
                        size=self._embedder.dense_size, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    _SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            client.create_payload_index(self._chunks, "doc_id", models.PayloadSchemaType.KEYWORD)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Returns the registry entry for doc_id, or None."""
        found = self._client.retrieve(self._documents, [_point_id(doc_id)])
        return DocumentRecord.model_validate(found[0].payload) if found else None

    def list_documents(self) -> list[DocumentRecord]:
        """Returns every registry entry."""
        points, _ = self._client.scroll(self._documents, limit=10_000)
        return sorted(
            (DocumentRecord.model_validate(p.payload) for p in points), key=lambda d: d.doc_id
        )

    def delete_document(self, doc_id: str) -> None:
        """Removes a document's chunks and its registry entry."""
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        )
        self._client.delete(self._chunks, points_selector=selector)
        self._client.delete(
            self._documents, points_selector=models.PointIdsList(points=[_point_id(doc_id)])
        )

    def upsert_document(self, document: chunker.SourceDocument) -> bool:
        """Indexes a document unless the same content hash is already stored.

        Args:
            document: Parsed document.

        Returns:
            True if the index changed, False if the document was unchanged.
        """
        current = self.get_document(document.doc_id)
        if current and current.content_hash == document.content_hash:
            return False
        self.delete_document(document.doc_id)
        vectors = self._embedder.embed_documents([chunk.text for chunk in document.chunks])
        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    _DENSE: dense,
                    _SPARSE: models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload=chunk.model_dump(),
            )
            for chunk, (dense, sparse) in zip(document.chunks, vectors, strict=True)
        ]
        self._client.upsert(self._chunks, points=points)
        record = DocumentRecord(
            doc_id=document.doc_id,
            title=document.title,
            version=document.version,
            updated=document.updated,
            content_hash=document.content_hash,
            chunk_count=len(points),
        )
        self._client.upsert(
            self._documents,
            points=[
                models.PointStruct(
                    id=_point_id(document.doc_id), vector={}, payload=record.model_dump()
                )
            ],
        )
        return True

    def index_version(self) -> str | None:
        """Version of the stored corpus, or None when nothing is indexed."""
        records = self.list_documents()
        if not records:
            return None
        return chunker.index_version({r.doc_id: r.content_hash for r in records})

    def search(self, query: str, limit: int) -> list[rag.RetrievedChunk]:
        """Hybrid search: dense and BM25 candidates fused with RRF, scored by cosine similarity.

        Args:
            query: User question.
            limit: Maximum number of chunks.

        Returns:
            Chunks in fused rank order, each with its dense cosine score in [0, 1].
        """
        dense, sparse = self._embedder.embed_query(query)
        candidates = max(limit * 4, 10)
        result = self._client.query_points(
            self._chunks,
            prefetch=[
                models.Prefetch(query=dense, using=_DENSE, limit=candidates),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=_SPARSE,
                    limit=candidates,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=[_DENSE],
        )
        return [
            rag.RetrievedChunk(
                doc_id=point.payload["doc_id"],
                title=point.payload["title"],
                chunk_id=point.payload["chunk_id"],
                text=point.payload["text"],
                score=cosine_score(dense, point.vector[_DENSE]),
            )
            for point in result.points
        ]
