"""Local admission control that rejects excess work with 429 instead of queueing it."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator

from app.core import errors


class RequestLimiter:
    """Caps in-flight requests and requests per rolling 60-second window.

    Cloud Run runs a single instance (D41), so a process-local limiter is the whole limit.
    Not thread-safe; it relies on the single asyncio event loop.

    Args:
        max_concurrent: Maximum requests in flight.
        per_minute: Maximum requests admitted per 60-second window.
        clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        max_concurrent: int,
        per_minute: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._per_minute = per_minute
        self._clock = clock
        self._in_flight = 0
        self._window_start = clock()
        self._window_count = 0

    @contextlib.contextmanager
    def acquire(self) -> Iterator[None]:
        """Admits one request for the duration of the block.

        Raises:
            errors.RateLimitedError: If either limit is exhausted.
        """
        now = self._clock()
        if now - self._window_start >= 60:
            self._window_start, self._window_count = now, 0
        if self._in_flight >= self._max_concurrent or self._window_count >= self._per_minute:
            raise errors.RateLimitedError("Too many requests; retry later.")
        self._in_flight += 1
        self._window_count += 1
        try:
            yield
        finally:
            self._in_flight -= 1
