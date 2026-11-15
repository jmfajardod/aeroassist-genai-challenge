"""Synchronizes the knowledge base into Qdrant; runnable as a CLI or a Cloud Run Job.

Usage: python -m app.retrieval.ingest
"""

from __future__ import annotations

import sys

import pydantic

from app.core import config
from app.retrieval import chunker, repository, retriever


class IngestReport(pydantic.BaseModel):
    """What one synchronization changed."""

    index_version: str | None
    changed: list[str]
    unchanged: list[str]
    deleted: list[str]


def sync(
    repo: repository.QdrantRepository, documents: list[chunker.SourceDocument]
) -> IngestReport:
    """Upserts changed documents and deletes indexed documents no longer in the corpus.

    Idempotent: a second run with the same corpus changes nothing.

    Args:
        repo: Target repository.
        documents: The full corpus.

    Returns:
        The report of changes.
    """
    changed, unchanged = [], []
    for document in documents:
        (changed if repo.upsert_document(document) else unchanged).append(document.doc_id)
    wanted = {d.doc_id for d in documents}
    deleted = [r.doc_id for r in repo.list_documents() if r.doc_id not in wanted]
    for doc_id in deleted:
        repo.delete_document(doc_id)
    return IngestReport(
        index_version=repo.index_version(), changed=changed, unchanged=unchanged, deleted=deleted
    )


def main() -> int:
    """Ingests the configured knowledge base into the configured Qdrant and prints the report."""
    settings = config.get_settings()
    report = sync(
        retriever.build_repository(settings), chunker.load_corpus(settings.knowledge_base_path)
    )
    print(report.model_dump_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
