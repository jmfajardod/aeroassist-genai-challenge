"""Dense and sparse text embeddings behind one interface: FastEmbed or a deterministic fake."""

from __future__ import annotations

import collections
import hashlib
import math
import pathlib
import re
from typing import Protocol

import pydantic

_TOKEN = re.compile(r"\w+", re.UNICODE)


class SparseVector(pydantic.BaseModel):
    """Sparse term weights, independent of the embedding library."""

    model_config = pydantic.ConfigDict(frozen=True)

    indices: list[int]
    values: list[float]


class Embedder(Protocol):
    """Produces one dense and one sparse vector per text."""

    @property
    def dense_size(self) -> int:
        """Dense vector dimension."""

    def embed_documents(self, texts: list[str]) -> list[tuple[list[float], SparseVector]]:
        """Embeds texts to be indexed.

        Args:
            texts: Chunk texts.

        Returns:
            One (dense, sparse) pair per text.
        """

    def embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        """Embeds a search query."""


class FastEmbedEmbedder:
    """ONNX models from FastEmbed: a multilingual dense model and BM25 with Spanish stemming.

    Args:
        dense_model: FastEmbed dense model name.
        sparse_model: FastEmbed sparse model name.
        cache_dir: Where model files live; baked into the image so nothing downloads at runtime.

    FastEmbed is imported on construction so the fake path never loads ONNX Runtime.
    """

    def __init__(
        self, dense_model: str, sparse_model: str, cache_dir: pathlib.Path | None = None
    ) -> None:
        import fastembed

        cache = str(cache_dir) if cache_dir else None
        self._dense = fastembed.TextEmbedding(dense_model, cache_dir=cache)
        self._sparse = fastembed.SparseTextEmbedding(
            sparse_model, cache_dir=cache, language="spanish"
        )
        self._size = len(next(iter(self._dense.embed(["dimension probe"]))))

    @property
    def dense_size(self) -> int:
        """Dense vector dimension."""
        return self._size

    def embed_documents(self, texts: list[str]) -> list[tuple[list[float], SparseVector]]:
        """Embeds texts to be indexed with the dense model and BM25 document weights."""
        dense = [vector.tolist() for vector in self._dense.embed(texts)]
        sparse = [
            SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
            for s in self._sparse.embed(texts)
        ]
        return list(zip(dense, sparse, strict=True))

    def embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        """Embeds a query with the dense model and BM25 query weights."""
        dense = next(iter(self._dense.query_embed(text))).tolist()
        sparse = next(iter(self._sparse.query_embed(text)))
        return dense, SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests and offline runs; no model files needed.

    Args:
        size: Dense vector dimension.
    """

    def __init__(self, size: int = 256) -> None:
        self._size = size

    @property
    def dense_size(self) -> int:
        """Dense vector dimension."""
        return self._size

    @staticmethod
    def _bucket(token: str, modulo: int) -> int:
        return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % modulo

    def _embed(self, text: str) -> tuple[list[float], SparseVector]:
        counts = collections.Counter(token.lower() for token in _TOKEN.findall(text))
        dense = [0.0] * self._size
        for token, count in counts.items():
            dense[self._bucket(token, self._size)] += count
        norm = math.sqrt(sum(v * v for v in dense)) or 1.0
        sparse: dict[int, float] = {}
        for token, count in counts.items():
            index = self._bucket(token, 2**20)
            sparse[index] = sparse.get(index, 0.0) + count
        indices = sorted(sparse)
        return [v / norm for v in dense], SparseVector(
            indices=indices, values=[sparse[i] for i in indices]
        )

    def embed_documents(self, texts: list[str]) -> list[tuple[list[float], SparseVector]]:
        """Embeds texts to be indexed."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        """Embeds a search query."""
        return self._embed(text)
