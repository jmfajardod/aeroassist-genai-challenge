"""Cost accounting, trace masking, LLM-input redaction and prompt fallback."""

from __future__ import annotations

import pytest

from app.api import readiness
from app.observability import redaction, tracing, usage
from app.orchestration import prompting


def test_cost_bills_cached_input_at_the_cached_rate() -> None:
    cost = usage.estimate_cost(
        "gpt-6.1-terra", input_tokens=1000, cached_tokens=400, output_tokens=200
    )
    assert cost == pytest.approx((600 * 1.00 + 400 * 0.05 + 200 * 5.00) / 1_000_000)
    assert usage.estimate_cost("unknown-model", 1000, 0, 200) is None


def test_tracker_totals_and_unknown_prices() -> None:
    tracker = usage.UsageTracker()
    tracker.record("classifier", "gpt-6.1-terra", {"input_tokens": 500, "output_tokens": 50})
    tracker.record("answer", "gpt-6-luna", None)
    summary = tracker.summary()
    assert summary is not None and summary.llm_calls == 1 and summary.output_tokens == 50
    tracker.record("tools", "unknown-model", {"input_tokens": 10, "output_tokens": 1})
    assert tracker.summary().cost_usd is None


class _Opaque:
    def __repr__(self) -> str:
        return "Send(node='retrieve', arg={'message': 'Reserva Q7M2PX'})"


def test_trace_mask_redacts_nested_values_and_fails_closed() -> None:
    data = {"input": [{"content": "Reserva Q7M2PX, correo ana@example.com"}], "n": 3}
    masked = tracing.mask(data=data)
    assert masked == {"input": [{"content": "Reserva [BOOKING], correo [EMAIL]"}], "n": 3}
    assert "Q7M2PX" not in str(tracing.mask(data=[_Opaque()]))


def test_llm_input_keeps_operational_identifiers() -> None:
    text = "Q7M2PX en LA2376 el 2026-08-06; llamar al +56 9 8765 4321 o ana@example.com"
    assert redaction.redact_personal(text) == (
        "Q7M2PX en LA2376 el 2026-08-06; llamar al [PHONE] o [EMAIL]"
    )


class _BrokenLangfuse:
    def get_prompt(self, *args: object, **kwargs: object) -> object:
        raise ConnectionError("down")


def test_prompt_falls_back_to_bundled_copy_when_langfuse_fails() -> None:
    prompt = prompting.load_from_langfuse(_BrokenLangfuse(), "production", "answer")
    assert prompt == prompting.load("answer")


async def test_readiness_does_not_cache_failures() -> None:
    state = {"ready": False}

    async def check() -> bool:
        return state["ready"]

    probe = readiness.Readiness({"index": check}, timeout_seconds=1, cache_seconds=60)
    assert await probe.run() == {"index": False}
    state["ready"] = True
    assert await probe.run() == {"index": True}
    state["ready"] = False
    assert await probe.run() == {"index": True}
