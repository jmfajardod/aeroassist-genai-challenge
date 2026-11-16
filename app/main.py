"""HTTP application exposing the public AeroAssist API contract."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
import time
from collections.abc import AsyncIterator

import fastapi
import prometheus_client
from fastapi import responses

from app import __version__, llm, schemas
from app import agent as agent_lib
from app.api import limits, readiness
from app.core import config, constants, errors
from app.mcp_tools import runner, transport
from app.observability import log, metrics, middleware, tracing, usage
from app.orchestration import prompting
from app.retrieval import chunker, ingest, retriever


class _Runtime:
    """Holds lazily built dependencies, since TestClient without with skips lifespan."""

    def __init__(self, settings: config.Settings, chat_agent: agent_lib.AeroAssistAgent | None):
        self.settings = settings
        self._agent = chat_agent
        self._transport: transport.McpTransport | None = None
        self._retriever: retriever.QdrantRetriever | None = None
        self._lock = threading.Lock()
        self.tracer = tracing.Tracer(settings)

    def transport(self) -> transport.McpTransport:
        """Returns the MCP transport; the subprocess starts on the first call."""
        if self._transport is None:
            self._transport = transport.StdioMcpTransport(
                self.settings.mcp_call_timeout_seconds, self.settings.mcp_startup_timeout_seconds
            )
        return self._transport

    def _build_agent(self) -> agent_lib.AeroAssistAgent:
        with self._lock:
            if self._agent is None:
                self._agent = agent_lib.AeroAssistAgent(
                    llm.build_llm(self.settings),
                    self.settings,
                    retriever=self._build_retriever(),
                    tool_runner=runner.ToolRunner(self.transport()),
                    tracer=self.tracer,
                    prompts=self._prompt_loader(),
                )
            return self._agent

    def _prompt_loader(self) -> prompting.PromptLoader:
        """Langfuse Prompt Management when configured and tracing is on, else bundled prompts."""
        if self.settings.prompt_source == "langfuse" and self.tracer.client is not None:
            return functools.partial(
                prompting.load_from_langfuse,
                self.tracer.client,
                self.settings.langfuse_prompt_label,
            )
        return prompting.load

    def _build_retriever(self) -> retriever.QdrantRetriever:
        """Connects to Qdrant; without QDRANT_URL it indexes the bundled corpus in process."""
        repo = retriever.build_repository(self.settings)
        if not self.settings.qdrant_url:
            ingest.sync(repo, chunker.load_corpus(self.settings.knowledge_base_path))
        self._retriever = retriever.QdrantRetriever(
            repo, self.settings.rag_min_score, self.settings.rag_evidence_score
        )
        return self._retriever

    async def agent(self) -> agent_lib.AeroAssistAgent:
        """Returns the chat agent, building it in a worker thread on first use.

        Raises:
            errors.NotReadyError: If the configured LLM backend cannot be built.
        """
        if self._agent is not None:
            return self._agent
        return await asyncio.to_thread(self._build_agent)

    async def llm_ready(self) -> bool:
        """Readiness check: the configured LLM backend can be built."""
        await self.agent()
        return True

    async def index_ready(self) -> bool:
        """Readiness check: Qdrant holds the index version of the bundled knowledge base."""
        await self.agent()
        if self._retriever is None:
            return False
        expected = await asyncio.to_thread(retriever.expected_index_version, self.settings)
        return self._retriever.index_version == expected

    async def warm_up(self) -> None:
        """Builds the agent, index and MCP session at startup; failures are left to readiness."""
        try:
            await self.agent()
            await self.transport().list_tools()
        except Exception as exc:
            log.logger.warning("warm_up_failed", extra={"error": type(exc).__name__})

    async def mcp_ready(self) -> bool:
        """Readiness check: the MCP server answers and exposes every allowlisted tool."""
        names = set(await self.transport().list_tools())
        return constants.TOOL_NAMES <= names


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
    log.configure_logging(settings.log_level)
    runtime = _Runtime(settings, agent)
    registry = metrics.Metrics()
    limiter = limits.RequestLimiter(
        settings.max_concurrent_requests, settings.rate_limit_per_minute
    )
    probe = readiness.Readiness(
        {"llm": runtime.llm_ready, "mcp": runtime.mcp_ready, "index": runtime.index_ready},
        timeout_seconds=settings.ready_check_timeout_seconds,
        cache_seconds=settings.ready_cache_seconds,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: fastapi.FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(runtime.warm_up())
        yield
        task.cancel()
        runtime.tracer.flush()

    application = fastapi.FastAPI(title="AeroAssist", version=__version__, lifespan=lifespan)
    errors.install_handlers(application)
    middleware.install_request_middleware(application, registry)

    @application.get("/health", response_model=schemas.HealthResponse)
    async def health() -> schemas.HealthResponse:
        return schemas.HealthResponse(version=__version__)

    @application.get("/ready", response_model=schemas.ReadyResponse)
    async def ready() -> responses.JSONResponse:
        checks = await probe.run()
        is_ready = bool(checks) and all(checks.values())
        body = schemas.ReadyResponse(status="ready" if is_ready else "not_ready", checks=checks)
        return responses.JSONResponse(
            status_code=200 if is_ready else 503, content=body.model_dump()
        )

    @application.get("/v1/capabilities", response_model=schemas.CapabilitiesResponse)
    async def capabilities() -> schemas.CapabilitiesResponse:
        variant = config.load_variant(settings.variant_path)
        return schemas.CapabilitiesResponse(
            challenge_version=variant.challenge_version,
            levels=schemas.CapabilityLevels(
                foundation=False, core=True, tools_mcp=True, rag=True, production=False
            ),
            llm_backend=settings.llm_backend,
            tool_transport="mcp-stdio",
            variant_id=variant.variant_id,
            reference_date=variant.reference_date,
        )

    @application.post("/v1/chat", response_model=schemas.ChatResponse)
    async def chat(payload: schemas.ChatRequest, request: fastapi.Request) -> schemas.ChatResponse:
        started = time.perf_counter()
        with limiter.acquire():
            try:
                async with asyncio.timeout(settings.request_deadline_seconds):
                    chat_agent = await runtime.agent()
                    tracker = usage.start()
                    response = await chat_agent.chat(payload, request_id=request.state.request_id)
            except TimeoutError as exc:
                raise errors.UpstreamTimeoutError("The request exceeded its time budget.") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        response = response.model_copy(
            update={"latency_ms": latency_ms, "usage": tracker.summary()}
        )
        registry.observe_chat(response, tracker.calls)
        return response

    @application.get("/metrics")
    async def metrics_endpoint() -> fastapi.Response:
        return fastapi.Response(
            content=registry.render(), media_type=prometheus_client.CONTENT_TYPE_LATEST
        )

    return application


app = create_app()
