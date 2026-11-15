"""Runtime configuration from environment variables and the challenge variant file."""

from __future__ import annotations

import datetime
import functools
import pathlib
from typing import Literal

import pydantic
import pydantic_settings

_ROOT = pathlib.Path(__file__).resolve().parents[2]

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]


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
    llm_base_url: str | None = None
    classifier_model: str = "gpt-6.1-terra"
    classifier_max_tokens: int = pydantic.Field(default=1024, gt=0)
    classifier_reasoning_effort: ReasoningEffort = "low"
    answer_model: str = "gpt-6.1-terra"
    answer_max_tokens: int = pydantic.Field(default=2048, gt=0)
    answer_reasoning_effort: ReasoningEffort = "none"
    tools_model: str = "gpt-6-luna"
    tools_max_tokens: int = pydantic.Field(default=1024, gt=0)
    tools_reasoning_effort: ReasoningEffort = "none"
    llm_timeout_seconds: float = pydantic.Field(default=8.0, gt=0)
    llm_max_retries: int = pydantic.Field(default=3, ge=0)
    llm_retry_backoff_seconds: float = pydantic.Field(default=1.0, ge=0)
    max_graph_steps: int = pydantic.Field(default=12, ge=4)
    max_tool_rounds: int = pydantic.Field(default=3, ge=1)
    max_tool_calls: int = pydantic.Field(default=6, ge=1)
    mcp_call_timeout_seconds: float = pydantic.Field(default=5.0, gt=0)
    mcp_startup_timeout_seconds: float = pydantic.Field(default=10.0, gt=0)
    request_deadline_seconds: float = pydantic.Field(default=35.0, gt=0, lt=40)
    max_concurrent_requests: int = pydantic.Field(default=10, ge=1)
    rate_limit_per_minute: int = pydantic.Field(default=60, ge=1)
    ready_check_timeout_seconds: float = pydantic.Field(default=2.0, gt=0)
    ready_cache_seconds: float = pydantic.Field(default=5.0, ge=0)
    variant_path: pathlib.Path = _ROOT / "config" / "challenge_variant.json"
    knowledge_base_path: pathlib.Path = _ROOT / "knowledge_base"
    qdrant_url: str | None = None
    qdrant_api_key: pydantic.SecretStr | None = None
    qdrant_timeout_seconds: float = pydantic.Field(default=5.0, gt=0)
    qdrant_collection_prefix: str = "aeroassist"
    embedding_backend: Literal["fastembed", "fake"] = "fastembed"
    dense_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    embedding_cache_dir: pathlib.Path | None = None
    rag_top_k: int = pydantic.Field(default=3, ge=1, le=10)
    rag_min_score: float = pydantic.Field(default=0.2, ge=0, le=1)
    rag_evidence_score: float = pydantic.Field(default=0.45, ge=0, le=1)


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
