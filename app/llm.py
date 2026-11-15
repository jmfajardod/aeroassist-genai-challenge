"""LLM provider boundary: the interface every backend implements and the backend factory."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

import pydantic

from app.core import config, errors
from app.orchestration import prompting
from app.providers import fake_llm, langchain_llm, outputs

SchemaT = TypeVar("SchemaT", bound=pydantic.BaseModel)


class LLMProvider(Protocol):
    """Interface every LLM backend implements; all outputs are typed."""

    @property
    def model_name(self) -> str:
        """Model ID reported in responses."""

    async def structured(
        self, *, role: outputs.Role, prompt: prompting.RenderedPrompt, schema: type[SchemaT]
    ) -> SchemaT:
        """Calls the role's model and returns its output parsed into schema.

        Args:
            role: Which configured model to use.
            prompt: Rendered system and user messages.
            schema: Pydantic output model.

        Returns:
            The parsed output.

        Raises:
            errors.UpstreamError: If the provider fails after retries.
            errors.UpstreamTimeoutError: If the last attempt timed out.
        """

    async def choose_tools(
        self,
        *,
        prompt: prompting.RenderedPrompt,
        rounds: list[outputs.ToolRound],
        tools: list[dict[str, Any]],
    ) -> outputs.ToolStep:
        """Asks the tools model for its next tool calls.

        Args:
            prompt: Rendered system and user messages.
            rounds: Earlier proposed calls with their results or validation errors.
            tools: Function specs the model may call.

        Returns:
            The proposed calls, or the model's text when it proposes none.
        """


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
    return langchain_llm.LangChainLLM(settings)
