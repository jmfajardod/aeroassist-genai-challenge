"""LLM provider boundary and the backend factory."""

from __future__ import annotations

from typing import Protocol

from app import config, errors


class LLMProvider(Protocol):
    """Minimal interface every LLM backend implements."""

    @property
    def model_name(self) -> str: ...

    async def generate(self, *, system: str, user: str) -> str: ...


class FakeLLM:
    """Deterministic offline backend for tests, CI and keyless local runs."""

    @property
    def model_name(self) -> str:
        """Identifier reported in responses."""
        return "fake-deterministic"

    async def generate(self, *, system: str, user: str) -> str:
        """Returns a fixed reply without any network call."""
        del system, user
        return "Respuesta determinista del backend fake."


def build_llm(settings: config.Settings) -> LLMProvider:
    """Creates the configured LLM backend.

    Args:
        settings: Application settings.

    Returns:
        The provider selected by LLM_BACKEND.

    Raises:
        errors.NotReadyError: If the selected backend is not available in this revision.
    """
    if settings.llm_backend == "fake":
        return FakeLLM()
    raise errors.NotReadyError(f"LLM backend '{settings.llm_backend}' is not available yet.")
