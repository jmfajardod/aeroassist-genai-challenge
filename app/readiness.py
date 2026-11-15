"""Bounded, cached dependency checks behind /ready."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping

logger = logging.getLogger("aeroassist")

Check = Callable[[], Awaitable[bool]]


class Readiness:
    """Runs named checks concurrently, each under a timeout, and caches the result briefly.

    Args:
        checks: Check name to async callable returning True when ready.
        timeout_seconds: Per-check time limit; a timeout counts as not ready.
        cache_seconds: How long a result is reused.
        clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        checks: Mapping[str, Check],
        *,
        timeout_seconds: float,
        cache_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._checks = dict(checks)
        self._timeout = timeout_seconds
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._cached: tuple[float, dict[str, bool]] | None = None

    async def _run_one(self, name: str, check: Check) -> bool:
        try:
            return bool(await asyncio.wait_for(check(), self._timeout))
        except Exception as exc:
            logger.warning(
                "readiness_check_failed", extra={"check": name, "error": type(exc).__name__}
            )
            return False

    async def run(self) -> dict[str, bool]:
        """Returns each check's status, reusing a fresh cached result.

        Returns:
            Mapping of check name to readiness.
        """
        now = self._clock()
        if self._cached and now - self._cached[0] < self._cache_seconds:
            return self._cached[1]
        names = list(self._checks)
        results = await asyncio.gather(*(self._run_one(n, self._checks[n]) for n in names))
        self._cached = (now, dict(zip(names, results, strict=True)))
        return self._cached[1]
