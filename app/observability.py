"""Structured JSON logging, Prometheus metrics and the request-ID middleware."""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable

import fastapi
import prometheus_client

from app import errors, redaction

logger = logging.getLogger("aeroassist")

_RESERVED_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
_INCOMING_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0)


class JsonFormatter(logging.Formatter):
    """Formats records as one JSON object per line with every string field redacted."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializes the record, including extra fields, as redacted JSON.

        Args:
            record: The log record.

        Returns:
            A single-line JSON string using Cloud Logging's severity key.
        """
        payload: dict[str, object] = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, datetime.UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": redaction.redact(record.getMessage()),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = redaction.redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = redaction.redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    """Routes the aeroassist logger to stdout as JSON.

    Args:
        level: Minimum level name, e.g. "INFO".
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


class Metrics:
    """Per-app Prometheus registry with bounded labels only."""

    def __init__(self) -> None:
        self.registry = prometheus_client.CollectorRegistry()
        self.requests = prometheus_client.Counter(
            "aeroassist_requests_total",
            "HTTP requests.",
            ["route", "method", "status_class"],
            registry=self.registry,
        )
        self.errors = prometheus_client.Counter(
            "aeroassist_errors_total",
            "HTTP responses with status >= 400.",
            ["route", "status"],
            registry=self.registry,
        )
        self.latency = prometheus_client.Histogram(
            "aeroassist_request_latency_seconds",
            "End-to-end HTTP latency.",
            ["route"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )

    def observe(self, route: str, method: str, status: int, seconds: float) -> None:
        """Records one finished request.

        Args:
            route: Route template, never the raw path.
            method: HTTP method.
            status: Response status code.
            seconds: Elapsed wall time.
        """
        self.requests.labels(route, method, f"{status // 100}xx").inc()
        if status >= 400:
            self.errors.labels(route, str(status)).inc()
        self.latency.labels(route).observe(seconds)

    def render(self) -> bytes:
        """Returns the registry in Prometheus text format."""
        return prometheus_client.generate_latest(self.registry)


def _route_label(request: fastapi.Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def install_request_middleware(application: fastapi.FastAPI, metrics: Metrics) -> None:
    """Adds request IDs, metrics, access logs and a last-resort 500 envelope.

    A valid incoming X-Request-ID is reused; otherwise a new one is generated. The ID is
    stored on request.state and echoed in the response header.

    Args:
        application: The FastAPI app to configure.
        metrics: Registry that receives one observation per request.
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
            logger.exception("unhandled_error", extra={"request_id": request_id})
            response = errors.error_response(
                500, "internal_error", "Internal server error.", request_id
            )
        elapsed = time.perf_counter() - started
        route = _route_label(request)
        metrics.observe(route, request.method, response.status_code, elapsed)
        response.headers["X-Request-ID"] = request_id
        logger.info(
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
