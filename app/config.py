"""Runtime configuration from environment variables and the challenge variant file."""

from __future__ import annotations

import datetime
import functools
import pathlib
from typing import Literal

import pydantic
import pydantic_settings

_ROOT = pathlib.Path(__file__).resolve().parent.parent

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class Variant(pydantic.BaseModel):
    """Challenge variant metadata: version, variant ID and reference date."""

    model_config = pydantic.ConfigDict(frozen=True)

    challenge_version: str
    variant_id: str
    reference_date: datetime.date
    synthetic_data: bool


class Settings(pydantic_settings.BaseSettings):
    """Application settings; each field is read from the upper-case env var of the same name."""

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True
    )

    app_env: str = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    llm_backend: Literal["fake", "openai"] = "fake"
    openai_api_key: pydantic.SecretStr | None = None
    classifier_model: str = "gpt-6.1-terra"
    classifier_max_tokens: int = pydantic.Field(default=1024, gt=0)
    classifier_reasoning_effort: ReasoningEffort = "low"
    answer_model: str = "gpt-6-luna"
    answer_max_tokens: int = pydantic.Field(default=2048, gt=0)
    answer_reasoning_effort: ReasoningEffort = "low"
    llm_timeout_seconds: float = pydantic.Field(default=8.0, gt=0)
    llm_max_retries: int = pydantic.Field(default=3, ge=0)
    llm_retry_backoff_seconds: float = pydantic.Field(default=1.0, ge=0)
    max_graph_steps: int = pydantic.Field(default=12, ge=4)
    request_deadline_seconds: float = pydantic.Field(default=35.0, gt=0, lt=40)
    max_concurrent_requests: int = pydantic.Field(default=4, ge=1)
    rate_limit_per_minute: int = pydantic.Field(default=60, ge=1)
    ready_check_timeout_seconds: float = pydantic.Field(default=2.0, gt=0)
    ready_cache_seconds: float = pydantic.Field(default=5.0, ge=0)
    variant_path: pathlib.Path = _ROOT / "config" / "challenge_variant.json"


@functools.cache
def get_settings() -> Settings:
    """Returns the process-wide settings, read once."""
    return Settings()


@functools.cache
def load_variant(path: pathlib.Path) -> Variant:
    """Reads and validates the challenge variant file.

    Args:
        path: Location of the variant JSON file.

    Returns:
        The parsed variant.
    """
    return Variant.model_validate_json(path.read_text(encoding="utf-8"))
