"""Request-ID, metrics and access-log middleware with a last-resort 500 envelope."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable

import fastapi

from app.core import errors
from app.observability import log, metrics

_INCOMING_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _route_label(request: fastapi.Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def install_request_middleware(application: fastapi.FastAPI, registry: metrics.Metrics) -> None:
    """Adds request IDs, metrics, access logs and a last-resort 500 envelope.

    A valid incoming X-Request-ID is reused; otherwise a new one is generated. The ID is
    stored on request.state and echoed in the response header.

    Args:
        application: The FastAPI app to configure.
        registry: Metrics that receive one observation per request.
    """

    @application.middleware("http")
    async def request_context(
        request: fastapi.Request,
        call_next: Callable[[fastapi.Request], Awaitable[fastapi.Response]],
    ) -> fastapi.Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _INCOMING_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.logger.exception("unhandled_error", extra={"request_id": request_id})
            response = errors.error_response(
                500, "internal_error", "Internal server error.", request_id
            )
        elapsed = time.perf_counter() - started
        route = _route_label(request)
        registry.observe(route, request.method, response.status_code, elapsed)
        response.headers["X-Request-ID"] = request_id
        log.logger.info(
            "request",
            extra={
                "request_id": request_id,
                "route": route,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": round(elapsed * 1000),
            },
        )
        return response
