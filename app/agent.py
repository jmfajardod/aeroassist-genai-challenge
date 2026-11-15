"""Application orchestration boundary: runs the answer graph and builds the HTTP response."""

from __future__ import annotations

from app import llm, rag, schemas
from app.core import config, errors
from app.mcp_tools import runner
from app.orchestration import graph


class AeroAssistAgent:
    """Answers chat requests through the compiled graph.

    Args:
        provider: LLM backend.
        settings: Application settings.
        retriever: Optional policy retriever.
        tool_runner: Optional MCP tool runner.
    """

    def __init__(
        self,
        provider: llm.LLMProvider,
        settings: config.Settings,
        retriever: rag.Retriever | None = None,
        tool_runner: runner.ToolRunner | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._retriever = retriever
        self._graph = graph.build(provider, settings, retriever, tool_runner)

    async def chat(self, request: schemas.ChatRequest, *, request_id: str) -> schemas.ChatResponse:
        """Builds the response for one chat request.

        Args:
            request: Validated chat request.
            request_id: ID echoed in the response.

        Returns:
            The structured chat response; latency_ms is set by the HTTP layer.

        Raises:
            errors.PromptInjectionError: If the classifier flags the message.
        """
        state = await self._graph.ainvoke(
            graph.State(message=request.message, confirm_action=request.confirm_action),
            config={"recursion_limit": self._settings.max_graph_steps},
        )
        outcome: graph.Outcome = state["outcome"]
        if outcome.route == "injection":
            raise errors.PromptInjectionError("The request was blocked by the safety policy.")
        return schemas.ChatResponse(
            answer=outcome.answer,
            citations=outcome.citations,
            tool_calls=outcome.tool_calls,
            requires_confirmation=outcome.requires_confirmation,
            request_id=request_id,
            model=self._provider.model_name,
            latency_ms=0,
            abstained=outcome.abstained,
            grounding=outcome.grounding,
            similarity=outcome.similarity,
            route=outcome.route,
            index_version=self._retriever.index_version if self._retriever else None,
            latency_breakdown_ms=state.get("timings_ms", {}),
        )
