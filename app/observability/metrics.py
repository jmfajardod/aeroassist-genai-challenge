"""Prometheus metrics with bounded labels only."""

from __future__ import annotations

import prometheus_client

from app import schemas
from app.observability import usage

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
        self.llm_tokens = prometheus_client.Counter(
            "aeroassist_llm_tokens_total",
            "LLM tokens by model and kind (input, cached, output).",
            ["model", "kind"],
            registry=self.registry,
        )
        self.llm_cost = prometheus_client.Counter(
            "aeroassist_llm_cost_usd_total",
            "Estimated LLM cost in USD by model.",
            ["model"],
            registry=self.registry,
        )
        self.tool_calls = prometheus_client.Counter(
            "aeroassist_tool_calls_total",
            "Executed tool calls by tool and status.",
            ["tool", "status"],
            registry=self.registry,
        )
        self.chat_routes = prometheus_client.Counter(
            "aeroassist_chat_routes_total",
            "Chat responses by route and abstention.",
            ["route", "abstained"],
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

    def observe_chat(self, response: schemas.ChatResponse, calls: list[usage.LlmCall]) -> None:
        """Records LLM usage, executed tools and the route of one chat response.

        Args:
            response: Final chat response.
            calls: LLM calls made for the request.
        """
        for call in calls:
            self.llm_tokens.labels(call.model, "input").inc(call.input_tokens)
            self.llm_tokens.labels(call.model, "cached").inc(call.cached_tokens)
            self.llm_tokens.labels(call.model, "output").inc(call.output_tokens)
            if call.cost_usd is not None:
                self.llm_cost.labels(call.model).inc(call.cost_usd)
        for trace in response.tool_calls:
            self.tool_calls.labels(trace.name, trace.status).inc()
        self.chat_routes.labels(response.route or "unknown", str(response.abstained).lower()).inc()

    def render(self) -> bytes:
        """Returns the registry in Prometheus text format."""
        return prometheus_client.generate_latest(self.registry)
