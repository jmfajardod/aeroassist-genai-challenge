"""Chunking, document CRUD with hash dedup, sync and the retrieval thresholds."""

from __future__ import annotations

import pathlib

import pytest
import qdrant_client

from app import rag
from app.retrieval import chunker, embeddings, ingest, repository, retriever

KB = pathlib.Path("knowledge_base")


@pytest.fixture
def repo() -> repository.QdrantRepository:
    client = qdrant_client.QdrantClient(location=":memory:")
    store = repository.QdrantRepository(client, embeddings.HashingEmbedder(), "test")
    store.ensure_collections()
    return store


def _chunk(doc_id: str, score: float) -> rag.RetrievedChunk:
    return rag.RetrievedChunk(
        doc_id=doc_id, title=doc_id, chunk_id=f"{doc_id}#x", text="", score=score
    )


def test_chunk_ids_follow_doc_id_and_section_slug() -> None:
    documents = {d.doc_id: d for d in chunker.load_corpus(KB)}
    assert len(documents) == 7
    assert [c.chunk_id for c in documents["IRROPS-001"].chunks] == [
        "IRROPS-001#cancelacion",
        "IRROPS-001#confirmacion",
        "IRROPS-001#causa-de-la-disrupcion",
    ]
    assert len(documents["BAGGAGE-001"].chunks) == 1


def test_long_sections_are_split_with_overlap(tmp_path: pathlib.Path) -> None:
    sentences = " ".join(f"Regla {i} de reacomodación para pasajeros afectados." for i in range(80))
    doc = tmp_path / "long.md"
    doc.write_text(
        f"---\ndoc_id: LONG-001\ntitle: Larga\n---\n\n# Larga\n\n## Reglas\n\n{sentences}\n"
    )
    chunks = chunker.parse_document(doc).chunks
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks[:2]] == ["LONG-001#reglas-1", "LONG-001#reglas-2"]
    first_tail = chunks[0].text.split()[-5:]
    assert " ".join(first_tail) in chunks[1].text
    assert all(len(c.text) <= chunker.CHUNK_SIZE + 40 for c in chunks)


def test_unchanged_document_is_skipped_and_changed_one_replaced(
    repo: repository.QdrantRepository,
) -> None:
    document = chunker.parse_document(KB / "irregular_operations.md")
    assert repo.upsert_document(document) is True
    assert repo.upsert_document(document) is False
    edited = document.model_copy(update={"content_hash": "new", "chunks": document.chunks[:1]})
    assert repo.upsert_document(edited) is True
    hits = repo.search("cancelación confirmación causa", limit=10)
    assert {hit.chunk_id for hit in hits} == {"IRROPS-001#cancelacion"}
    assert repo.get_document("IRROPS-001").chunk_count == 1


def test_sync_is_idempotent_and_removes_deleted_documents(
    repo: repository.QdrantRepository,
) -> None:
    corpus = chunker.load_corpus(KB)
    first = ingest.sync(repo, corpus)
    assert len(first.changed) == 7
    second = ingest.sync(repo, corpus)
    assert second.changed == [] and second.index_version == first.index_version
    third = ingest.sync(repo, corpus[1:])
    assert third.deleted == [corpus[0].doc_id]
    assert third.index_version != first.index_version
    assert repo.search(corpus[0].chunks[0].text, limit=10)[0].doc_id != corpus[0].doc_id


def test_search_scores_are_bounded(repo: repository.QdrantRepository) -> None:
    ingest.sync(repo, chunker.load_corpus(KB))
    hits = repo.search("¿Se mantiene el equipaje incluido al reacomodar?", limit=3)
    assert hits and all(0.0 <= hit.score <= 1.0 for hit in hits)
    assert hits[0].doc_id == "BAGGAGE-001"


def test_evidence_gate_then_floor(repo: repository.QdrantRepository) -> None:
    gate = retriever.QdrantRetriever(repo, min_score=0.2, evidence_score=0.45)
    assert gate.apply_thresholds([_chunk("A", 0.44), _chunk("B", 0.3)]) == []
    kept = gate.apply_thresholds([_chunk("A", 0.5), _chunk("B", 0.25), _chunk("C", 0.1)])
    assert [c.doc_id for c in kept] == ["A", "B"]
