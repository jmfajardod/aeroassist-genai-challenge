"""Tool-argument validation, provenance, relative dates and idempotency keys."""

from __future__ import annotations

import datetime
from typing import Any

import pytest

from app.core import config, constants
from app.mcp_tools import contracts, provenance, runner, write_gate
from app.orchestration import graph, prompting
from app.providers import outputs

HOLD = constants.WRITE_TOOL


def test_derived_idempotency_key_is_stable_per_target() -> None:
    first = write_gate.derive_idempotency_key("Q7M2PX", "LA2376", "2026-08-06")
    assert first == write_gate.derive_idempotency_key("Q7M2PX", "LA2376", "2026-08-06")
    assert first != write_gate.derive_idempotency_key("Q7M2PX", "LA2696", "2026-08-06")


@pytest.mark.parametrize(
    ("name", "args"),
    [
        (HOLD, {"booking_code": "Q7M2PX", "flight_number": "LA2376", "idempotency_key": "x"}),
        ("get_flight_status", {"flight_number": "LA401; DROP", "departure_date": "2026-08-06"}),
        ("get_flight_status", {"flight_number": "LA401", "departure_date": "06/08/2026"}),
        ("get_booking_context", {"booking_code": "Q7M2PX", "extra": "1"}),
        ("delete_everything", {}),
    ],
)
def test_model_proposed_calls_are_validated(name: str, args: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        runner.validate_read(name, args)


def test_search_results_are_clamped() -> None:
    args = contracts.SearchArgs(
        origin="scl", destination="lim", departure_date="2026-08-06", max_results=50
    )
    assert (args.origin, args.max_results) == ("SCL", 5)


class _HallucinatingLLM:
    """Proposes a status call with a date the user never gave."""

    @property
    def model_name(self) -> str:
        return "scripted"

    async def choose_tools(
        self, *, prompt: prompting.RenderedPrompt, rounds: list[Any], tools: list[Any]
    ) -> outputs.ToolStep:
        if rounds:
            return outputs.ToolStep(requests=[], text="¿Para qué fecha?")
        request = outputs.ToolRequest(
            id="c1",
            name="get_flight_status",
            arguments={"flight_number": "LA401", "departure_date": "2026-08-06"},
        )
        return outputs.ToolStep(requests=[request])


class _RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        return {"found": True}

    async def list_tools(self) -> list[str]:
        return []


async def test_arguments_without_provenance_never_run() -> None:
    recorder = _RecordingTransport()
    update = await graph.gather_tool_data(
        _HallucinatingLLM(),
        runner.ToolRunner(recorder),
        config.Settings(_env_file=None),
        "¿Cuál es el estado del vuelo LA401?",
        datetime.date(2026, 8, 6),
    )
    assert recorder.calls == []
    assert update["tool_calls"] == []
    assert update["clarification"] == "¿Para qué fecha?"


def test_relative_dates_resolve_against_reference_date() -> None:
    assert provenance.resolve_dates("LA500 de mañana", datetime.date(2026, 8, 6)) == {"2026-08-07"}
