"""LangChain chat-model backend with per-call timeouts and bounded retries."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import openai
import pydantic
from langchain import chat_models
from langchain_core import messages as lc_messages

from app.core import config, errors
from app.orchestration import prompting
from app.providers import outputs

logger = logging.getLogger("aeroassist")

SchemaT = TypeVar("SchemaT", bound=pydantic.BaseModel)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def is_retryable(exc: BaseException) -> bool:
    """True for timeouts, connection failures, throttling and provider 5xx errors."""
    if isinstance(exc, TimeoutError | openai.APIConnectionError | openai.RateLimitError):
        return True
    return getattr(exc, "status_code", None) in _RETRYABLE_STATUS


async def call_with_retries(
    call: Callable[[], Awaitable[SchemaT]],
    *,
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
) -> SchemaT:
    """Runs a provider call under a per-attempt timeout, retrying retryable failures.

    The request-level deadline cancels pending attempts and backoff sleeps.

    Args:
        call: Zero-argument coroutine factory, invoked once per attempt.
        timeout_seconds: Limit for each attempt.
        max_retries: Retries after the first attempt.
        backoff_seconds: Fixed wait between attempts.

    Returns:
        The call result.

    Raises:
        errors.UpstreamTimeoutError: If the last attempt timed out.
        errors.UpstreamError: If the provider failed in any other way.
    """
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(call(), timeout_seconds)
        except Exception as exc:
            retry = is_retryable(exc) and attempt < max_retries
            logger.warning(
                "llm_call_failed",
                extra={"attempt": attempt + 1, "error": type(exc).__name__, "retry": retry},
            )
            if retry:
                await asyncio.sleep(backoff_seconds)
                continue
            if isinstance(exc, TimeoutError | openai.APITimeoutError):
                raise errors.UpstreamTimeoutError("The language model timed out.") from exc
            raise errors.UpstreamError("The language model provider failed.") from exc
    raise AssertionError("unreachable")


class LangChainLLM:
    """Provider-agnostic adapter over LangChain chat models, one model per role.

    Args:
        settings: Application settings with per-role model, token and reasoning limits.
    """

    def __init__(self, settings: config.Settings) -> None:
        self._settings = settings
        self._models = {
            "classifier": self._chat_model(
                settings.classifier_model,
                settings.classifier_max_tokens,
                settings.classifier_reasoning_effort,
            ),
            "answer": self._chat_model(
                settings.answer_model, settings.answer_max_tokens, settings.answer_reasoning_effort
            ),
            "tools": self._chat_model(
                settings.tools_model, settings.tools_max_tokens, settings.tools_reasoning_effort
            ),
        }
        self._runnables: dict[tuple[str, type], Any] = {}

    def _chat_model(self, model: str, max_tokens: int, reasoning_effort: str) -> Any:
        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "max_retries": 0}
        if self._settings.llm_backend == "openai":
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["api_key"] = self._settings.openai_api_key
            kwargs["timeout"] = self._settings.llm_timeout_seconds
        if self._settings.llm_base_url:
            kwargs["base_url"] = self._settings.llm_base_url
        return chat_models.init_chat_model(
            model, model_provider=self._settings.llm_backend, **kwargs
        )

    @property
    def model_name(self) -> str:
        """Answer-generation model ID reported in responses."""
        return self._settings.answer_model

    async def structured(
        self, *, role: outputs.Role, prompt: prompting.RenderedPrompt, schema: type[SchemaT]
    ) -> SchemaT:
        """Calls the role's model and parses the reply into schema.

        Args:
            role: Which configured model to use.
            prompt: Rendered system and user messages.
            schema: Output model; decoded with strict JSON schema.

        Returns:
            The parsed output.
        """
        key = (role, schema)
        if key not in self._runnables:
            self._runnables[key] = self._models[role].with_structured_output(
                schema, method="json_schema", strict=True
            )
        runnable = self._runnables[key]
        messages = [("system", prompt.system), ("human", prompt.user)]
        return await call_with_retries(
            lambda: runnable.ainvoke(messages),
            timeout_seconds=self._settings.llm_timeout_seconds,
            max_retries=self._settings.llm_max_retries,
            backoff_seconds=self._settings.llm_retry_backoff_seconds,
        )

    async def choose_tools(
        self,
        *,
        prompt: prompting.RenderedPrompt,
        rounds: list[outputs.ToolRound],
        tools: list[dict[str, Any]],
    ) -> outputs.ToolStep:
        """Asks the tools model for its next tool calls given the results so far.

        Args:
            prompt: Rendered system and user messages.
            rounds: Earlier tool calls with their results (or validation errors).
            tools: Function specs the model may call.

        Returns:
            The proposed calls, or the model's text when it proposes none.
        """
        history: list[lc_messages.BaseMessage] = [
            lc_messages.SystemMessage(prompt.system),
            lc_messages.HumanMessage(prompt.user),
        ]
        for tool_round in rounds:
            history.append(
                lc_messages.AIMessage(
                    content="",
                    tool_calls=[
                        {"id": req.id, "name": req.name, "args": req.arguments}
                        for req, _ in tool_round
                    ],
                )
            )
            history.extend(
                lc_messages.ToolMessage(json.dumps(result, ensure_ascii=False), tool_call_id=req.id)
                for req, result in tool_round
            )
        model = self._models["tools"].bind_tools(tools)
        reply = await call_with_retries(
            lambda: model.ainvoke(history),
            timeout_seconds=self._settings.llm_timeout_seconds,
            max_retries=self._settings.llm_max_retries,
            backoff_seconds=self._settings.llm_retry_backoff_seconds,
        )
        requests = [
            outputs.ToolRequest(
                id=call["id"] or f"call-{i}", name=call["name"], arguments=call["args"]
            )
            for i, call in enumerate(reply.tool_calls)
        ]
        text = str(reply.text) if not requests else ""
        return outputs.ToolStep(requests=requests, text=text)
