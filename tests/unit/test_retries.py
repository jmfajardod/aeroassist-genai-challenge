"""Per-call timeout and bounded retries around provider calls."""

from __future__ import annotations

import pytest

from app.core import errors
from app.providers import langchain_llm


class _Flaky:
    def __init__(self, failures: list[Exception]) -> None:
        self.failures = failures
        self.attempts = 0

    async def __call__(self) -> str:
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


async def test_retries_retryable_errors_then_succeeds() -> None:
    call = _Flaky([TimeoutError(), ConnectionResetError()])
    call.failures[1].status_code = 503
    result = await langchain_llm.call_with_retries(
        call, timeout_seconds=1, max_retries=3, backoff_seconds=0
    )
    assert result == "ok" and call.attempts == 3


async def test_non_retryable_error_fails_fast_as_502() -> None:
    call = _Flaky([ValueError("bad schema")])
    with pytest.raises(errors.UpstreamError):
        await langchain_llm.call_with_retries(
            call, timeout_seconds=1, max_retries=3, backoff_seconds=0
        )
    assert call.attempts == 1


async def test_exhausted_timeouts_raise_504() -> None:
    call = _Flaky([TimeoutError()] * 4)
    with pytest.raises(errors.UpstreamTimeoutError):
        await langchain_llm.call_with_retries(
            call, timeout_seconds=1, max_retries=3, backoff_seconds=0
        )
    assert call.attempts == 4
