"""Parses knowledge-base markdown into documents and section chunks."""

from __future__ import annotations

import hashlib
import pathlib
import re
import unicodedata

import langchain_text_splitters
import pydantic

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
_HEADERS = langchain_text_splitters.MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True
)
_SPLITTER = langchain_text_splitters.RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " "]
)


class Chunk(pydantic.BaseModel):
    """One retrievable section of a document."""

    model_config = pydantic.ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    title: str
    text: str


class SourceDocument(pydantic.BaseModel):
    """A knowledge-base document with its metadata, content hash and chunks."""

    model_config = pydantic.ConfigDict(frozen=True)

    doc_id: str
    title: str
    version: str
    updated: str
    content_hash: str
    chunks: list[Chunk]


def slugify(text: str) -> str:
    """Lowercase ASCII slug, e.g. "Cancelación" becomes "cancelacion"."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def _clean(text: str) -> str:
    lines = (line.rstrip() for line in text.splitlines() if not line.startswith(">"))
    return "\n".join(lines).strip()


def parse_document(path: pathlib.Path) -> SourceDocument:
    """Splits one markdown file into chunks with LangChain's markdown and recursive splitters.

    Sections (H2) are the primary unit. A section longer than CHUNK_SIZE characters is split
    further with CHUNK_OVERLAP characters of overlap; its parts get numbered chunk_ids. A document
    without H2 sections becomes chunks named after its title. Disclaimer blockquotes are not
    indexed, and each chunk starts with the document title and heading so it is self-describing.

    Args:
        path: Markdown file with front matter.

    Returns:
        The parsed document.

    Raises:
        ValueError: If the front matter is missing or lacks doc_id.
    """
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise ValueError(f"{path.name}: missing front matter")
    meta = dict(
        (key.strip(), value.strip())
        for key, value in (
            line.split(":", 1) for line in match.group(1).splitlines() if ":" in line
        )
    )
    doc_id = meta["doc_id"]
    sections = _HEADERS.split_text(raw[match.end() :])
    title = meta.get("title") or next(
        (s.metadata["title"] for s in sections if "title" in s.metadata), path.stem
    )
    chunks = []
    for section in sections:
        heading = section.metadata.get("section", title)
        parts = _SPLITTER.split_text(_clean(section.page_content))
        for index, part in enumerate(parts, 1):
            suffix = f"-{index}" if len(parts) > 1 else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{slugify(heading)}{suffix}",
                    doc_id=doc_id,
                    title=title,
                    text=f"{title}. {heading}\n{part}",
                )
            )
    return SourceDocument(
        doc_id=doc_id,
        title=title,
        version=meta.get("version", ""),
        updated=meta.get("updated", ""),
        content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        chunks=chunks,
    )


def load_corpus(directory: pathlib.Path) -> list[SourceDocument]:
    """Parses every markdown file in a directory, sorted by doc_id."""
    return sorted((parse_document(p) for p in directory.glob("*.md")), key=lambda d: d.doc_id)


def index_version(hashes: dict[str, str]) -> str:
    """Short hash identifying a corpus state from each document's content hash.

    Args:
        hashes: doc_id to content hash.

    Returns:
        A 12-character version, equal for equal corpora.
    """
    joined = "\n".join(f"{doc_id}:{digest}" for doc_id, digest in sorted(hashes.items()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
