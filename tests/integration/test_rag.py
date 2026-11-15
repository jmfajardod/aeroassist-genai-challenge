"""RAG through the API: cited policy answers and index readiness."""

from __future__ import annotations

import asyncio
import pathlib
import shutil

import httpx
import pytest
from fastapi import testclient

from app import main
from app.core import config
from app.retrieval import retriever


def _client(**overrides: object) -> testclient.TestClient:
    settings = config.Settings(_env_file=None, **overrides)
    return testclient.TestClient(main.create_app(settings=settings))


def test_policy_answer_cites_retrieved_document() -> None:
    message = "Si me reacomodan por una cancelación, ¿mantengo el equipaje ya incluido?"
    client = _client(rag_min_score=0.0, rag_evidence_score=0.0)
    body = client.post("/v1/chat", json={"message": message}).json()
    assert "BAGGAGE-001" in {c["doc_id"] for c in body["citations"]}
    assert all(0.0 <= c["score"] <= 1.0 for c in body["citations"])
    assert body["index_version"]


def test_ready_is_503_when_the_index_is_stale(tmp_path: pathlib.Path) -> None:
    kb = tmp_path / "kb"
    shutil.copytree("knowledge_base", kb)
    client = _client(knowledge_base_path=kb, ready_cache_seconds=0)
    assert client.get("/ready").json()["checks"]["index"] is True
    (kb / "baggage.md").write_text((kb / "baggage.md").read_text() + "\nCambio.\n")
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["index"] is False


async def test_models_and_index_load_once_under_concurrent_first_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds = []
    original = retriever.build_repository

    def counting(settings: config.Settings) -> object:
        builds.append(settings)
        return original(settings)

    monkeypatch.setattr(retriever, "build_repository", counting)
    app = main.create_app(settings=config.Settings(_env_file=None))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        replies = await asyncio.gather(
            *(
                client.post("/v1/chat", json={"message": "¿Equipaje tras reacomodación?"})
                for _ in range(5)
            )
        )
    assert [reply.status_code for reply in replies] == [200] * 5
    assert len(builds) == 1
