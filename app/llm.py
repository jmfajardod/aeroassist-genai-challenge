"""LLM provider boundary: structured-output calls with timeouts and bounded retries."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, TypeVar

import openai
import pydantic
from langchain import chat_models

from app import config, errors, fake_llm, prompting

logger = logging.getLogger("aeroassist")

Role = Literal["classifier", "answer"]
SchemaT = TypeVar("SchemaT", bound=pydantic.BaseModel)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class LLMProvider(Protocol):
    """Interface every LLM backend implements; all outputs are pydantic models."""

    @property
    def model_name(self) -> str: ...

    async def structured(
        self, *, role: Role, prompt: prompting.RenderedPrompt, schema: type[SchemaT]
    ) -> SchemaT: ...


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
        }
        self._runnables: dict[tuple[str, type], Any] = {}

    def _chat_model(self, model: str, max_tokens: int, reasoning_effort: str) -> Any:
        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "max_retries": 0}
        if self._settings.llm_backend == "openai":
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["api_key"] = self._settings.openai_api_key
            kwargs["timeout"] = self._settings.llm_timeout_seconds
        return chat_models.init_chat_model(
            model, model_provider=self._settings.llm_backend, **kwargs
        )

    @property
    def model_name(self) -> str:
        """Answer-generation model ID reported in responses."""
        return self._settings.answer_model

    async def structured(
        self, *, role: Role, prompt: prompting.RenderedPrompt, schema: type[SchemaT]
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


def build_llm(settings: config.Settings) -> LLMProvider:
    """Creates the configured LLM backend.

    Args:
        settings: Application settings.

    Returns:
        The provider selected by LLM_BACKEND.

    Raises:
        errors.NotReadyError: If the backend needs an API key that is not configured.
    """
    if settings.llm_backend == "fake":
        return fake_llm.FakeLLM()
    if settings.openai_api_key is None:
        raise errors.NotReadyError("The LLM provider is not configured.")
    return LangChainLLM(settings)
