"""Prometheus metrics with bounded labels only."""

from __future__ import annotations

import prometheus_client

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0)


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
