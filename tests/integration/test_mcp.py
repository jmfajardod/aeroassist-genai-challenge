"""Real MCP stdio integration and the write gate end to end through the API."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import testclient

from app import main
from app.core import config, constants
from app.mcp_tools import transport

HOLD = constants.WRITE_TOOL


@pytest.fixture(scope="module")
def client() -> testclient.TestClient:
    return testclient.TestClient(main.create_app(settings=config.Settings(_env_file=None)))


def _chat(client: testclient.TestClient, message: str, confirm: bool = False) -> Any:
    return client.post("/v1/chat", json={"message": message, "confirm_action": confirm})


def _names(body: dict[str, Any]) -> list[str]:
    return [call["name"] for call in body["tool_calls"]]


async def test_stdio_round_trip_reads_synthetic_data() -> None:
    stdio = transport.StdioMcpTransport(call_timeout_seconds=5, startup_timeout_seconds=10)
    assert HOLD in await stdio.list_tools()
    result = await stdio.call_tool(
        "get_flight_status", {"flight_number": "LA401", "departure_date": "2026-08-06"}
    )
    assert result["status"] == "CANCELLED"


def test_confirmed_valid_hold_is_created_once(client: testclient.TestClient) -> None:
    body = _chat(client, "Confirmo el hold para Q7M2PX en LA2376 con clave t-1.", True).json()
    assert _names(body).count(HOLD) == 1
    assert body["requires_confirmation"] is False
    assert "HOLD-" in body["answer"]
    hold = next(c for c in body["tool_calls"] if c["name"] == HOLD)
    assert hold["arguments"]["idempotency_key"] == "t-1"


def test_hold_mentions_special_assistance(client: testclient.TestClient) -> None:
    body = _chat(client, "Crea el hold para Z3TRIP en LA532.").json()
    assert body["requires_confirmation"] is True
    assert HOLD not in _names(body)
    assert "WCHR" in body["answer"]


def test_target_outside_route_is_refused_without_write(client: testclient.TestClient) -> None:
    response = _chat(client, "Confirmo el hold para Q7M2PX en LA532 con clave t-2.", True)
    assert response.status_code == 200
    assert HOLD not in _names(response.json())
    assert response.json()["requires_confirmation"] is False


def test_unknown_booking_is_explained_without_write(client: testclient.TestClient) -> None:
    response = _chat(client, "Confirmo el hold para AAAAA1 en LA2376 con clave t-3.", True)
    assert response.status_code == 200
    assert HOLD not in _names(response.json())


def test_confirmed_write_without_booking_is_400(client: testclient.TestClient) -> None:
    response = _chat(client, "Confirmo el hold en LA2376 con clave t-4.", True)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsafe_request"


def test_two_targets_never_write(client: testclient.TestClient) -> None:
    body = _chat(client, "Confirmo holds para Q7M2PX en LA2376 y también en LA2696.", True).json()
    assert HOLD not in _names(body)
    assert body["requires_confirmation"] is True
