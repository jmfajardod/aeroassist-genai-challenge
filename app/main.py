"""HTTP application exposing the public AeroAssist API contract."""

from __future__ import annotations

import asyncio
import time

import fastapi
import prometheus_client
from fastapi import responses

from app import __version__, config, errors, limits, llm, observability, readiness, schemas
from app import agent as agent_lib


class _Runtime:
    """Holds lazily built dependencies, since TestClient without with skips lifespan."""

    def __init__(self, settings: config.Settings, chat_agent: agent_lib.AeroAssistAgent | None):
        self.settings = settings
        self._agent = chat_agent

    def agent(self) -> agent_lib.AeroAssistAgent:
        """Returns the chat agent, building it on first use.

        Raises:
            errors.NotReadyError: If the configured LLM backend cannot be built.
        """
        if self._agent is None:
            self._agent = agent_lib.AeroAssistAgent(llm.build_llm(self.settings), self.settings)
        return self._agent

    async def llm_ready(self) -> bool:
        """Readiness check: the configured LLM backend can be built."""
        self.agent()
        return True


def create_app(
    agent: agent_lib.AeroAssistAgent | None = None, settings: config.Settings | None = None
) -> fastapi.FastAPI:
    """Builds the FastAPI app with no I/O at import time.

    Args:
        agent: Optional pre-built agent, mainly for tests.
        settings: Optional settings; defaults to the environment.

    Returns:
        The configured application.
    """
    settings = settings or config.get_settings()
    observability.configure_logging(settings.log_level)
    runtime = _Runtime(settings, agent)
    metrics = observability.Metrics()
    limiter = limits.RequestLimiter(
        settings.max_concurrent_requests, settings.rate_limit_per_minute
    )
    probe = readiness.Readiness(
        {"llm": runtime.llm_ready},
        timeout_seconds=settings.ready_check_timeout_seconds,
        cache_seconds=settings.ready_cache_seconds,
    )

    application = fastapi.FastAPI(title="AeroAssist", version=__version__)
    errors.install_handlers(application)
    observability.install_request_middleware(application, metrics)

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @application.get("/ready")
    async def ready() -> responses.JSONResponse:
        checks = await probe.run()
        is_ready = bool(checks) and all(checks.values())
        return responses.JSONResponse(
            status_code=200 if is_ready else 503,
            content={"status": "ready" if is_ready else "not_ready", "checks": checks},
        )

    @application.get("/v1/capabilities", response_model=schemas.CapabilitiesResponse)
    async def capabilities() -> schemas.CapabilitiesResponse:
        variant = config.load_variant(settings.variant_path)
        return schemas.CapabilitiesResponse(
            challenge_version=variant.challenge_version,
            levels=schemas.CapabilityLevels(
                foundation=False, core=False, tools_mcp=False, rag=False, production=False
            ),
            llm_backend=settings.llm_backend,
            tool_transport="none",
            variant_id=variant.variant_id,
            reference_date=variant.reference_date,
        )

    @application.post("/v1/chat", response_model=schemas.ChatResponse)
    async def chat(payload: schemas.ChatRequest, request: fastapi.Request) -> schemas.ChatResponse:
        started = time.perf_counter()
        with limiter.acquire():
            try:
                async with asyncio.timeout(settings.request_deadline_seconds):
                    response = await runtime.agent().chat(
                        payload, request_id=request.state.request_id
                    )
            except TimeoutError as exc:
                raise errors.UpstreamTimeoutError("The request exceeded its time budget.") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        return response.model_copy(update={"latency_ms": latency_ms})

    @application.get("/metrics")
    async def metrics_endpoint() -> fastapi.Response:
        return fastapi.Response(
            content=metrics.render(), media_type=prometheus_client.CONTENT_TYPE_LATEST
        )

    return application


app = create_app()
