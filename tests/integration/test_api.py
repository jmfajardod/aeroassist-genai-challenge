"""Contract behavior beyond the public tests: errors, limits, readiness and metrics."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import testclient

from app import main, schemas
from app.core import config, errors

HELP = {"message": "¿Qué puedes hacer?"}


def _settings(**overrides: object) -> config.Settings:
    return config.Settings(_env_file=None, **overrides)


def _client(agent: object = None, **overrides: object) -> testclient.TestClient:
    app = main.create_app(agent, settings=_settings(**overrides))
    return testclient.TestClient(app, raise_server_exceptions=False)


class _RaisingAgent:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def chat(self, request: schemas.ChatRequest, *, request_id: str) -> None:
        raise self._exc


class _SlowAgent:
    async def chat(self, request: schemas.ChatRequest, *, request_id: str) -> None:
        await asyncio.sleep(5)


def test_validation_error_uses_envelope_without_echoing_input() -> None:
    response = _client().post("/v1/chat", json={"message": "ana@example.com " + "x" * 4000})
    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "validation_error"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert "ana@example.com" not in response.text


def test_length_limit_applies_after_trimming() -> None:
    response = _client().post("/v1/chat", json={"message": "   " + "a" * 4000 + "   "})
    assert response.status_code == 200


def test_unknown_fields_are_rejected() -> None:
    response = _client().post("/v1/chat", json={**HELP, "role": "system"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (errors.UpstreamError("LLM failed."), 502, "upstream_error"),
        (errors.PromptInjectionError("Blocked."), 400, "prompt_injection"),
        (errors.NotReadyError("Not ready."), 503, "not_ready"),
    ],
)
def test_app_errors_map_to_contract_status(exc: errors.AppError, status: int, code: str) -> None:
    response = _client(_RaisingAgent(exc)).post("/v1/chat", json=HELP)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_unexpected_error_returns_500_envelope_without_details() -> None:
    response = _client(_RaisingAgent(RuntimeError("sk-secret-key"))).post("/v1/chat", json=HELP)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "sk-secret-key" not in response.text


def test_deadline_exceeded_returns_504() -> None:
    client = _client(_SlowAgent(), request_deadline_seconds=0.05)
    response = client.post("/v1/chat", json=HELP)
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout"


def test_rate_limit_returns_429() -> None:
    client = _client(rate_limit_per_minute=1)
    assert client.post("/v1/chat", json=HELP).status_code == 200
    response = client.post("/v1/chat", json=HELP)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"


def test_ready_is_200_with_fake_backend() -> None:
    response = _client().get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_request_id_header_reuses_valid_id_and_replaces_unsafe_one() -> None:
    client = _client()
    valid = client.get("/health", headers={"X-Request-ID": "trace-12345678"})
    unsafe = client.get("/health", headers={"X-Request-ID": "Q7M2PX\nforged log line"})
    assert valid.headers["X-Request-ID"] == "trace-12345678"
    assert "Q7M2PX" not in unsafe.headers["X-Request-ID"]


def test_metrics_use_route_templates_not_raw_paths() -> None:
    client = _client()
    client.post("/v1/chat", json=HELP)
    client.get("/bookings/Q7M2PX")
    text = client.get("/metrics").text
    assert 'aeroassist_request_latency_seconds_count{route="/v1/chat"} 1.0' in text
    assert 'route="unmatched"' in text
    assert "Q7M2PX" not in text


def test_out_of_scope_question_abstains() -> None:
    body = _client().post("/v1/chat", json={"message": "¿Precio del dólar mañana?"}).json()
    assert body["abstained"] is True
    assert body["tool_calls"] == [] and body["requires_confirmation"] is False
