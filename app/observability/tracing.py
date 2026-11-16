"""Langfuse tracing through the LangChain callback, with PII masking and fail-open behavior."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import langfuse
from langfuse import langchain as langfuse_langchain

from app import __version__
from app.core import config
from app.observability import redaction

logger = logging.getLogger("aeroassist")


def mask(*, data: Any, **kwargs: Any) -> Any:
    """Langfuse mask hook: redacts PII in every trace input, output and metadata value."""
    del kwargs
    return redaction.redact_value(data)


class Tracer:
    """Owns the Langfuse client; every method is a no-op when tracing is off.

    Tracing is off when keys are missing or the client cannot be created, so a Langfuse outage
    never fails a request. Export is asynchronous in the SDK's background thread.

    Args:
        settings: Application settings with the Langfuse keys and host.
    """

    def __init__(self, settings: config.Settings) -> None:
        self.client: langfuse.Langfuse | None = None
        self._public_key = settings.langfuse_public_key
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return
        try:
            self.client = langfuse.Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url,
                mask=mask,
                environment=settings.app_env,
                release=__version__,
            )
        except Exception as exc:
            logger.warning("tracing_disabled", extra={"error": type(exc).__name__})

    def run_config(self, request_id: str, session_id: str | None) -> dict[str, Any]:
        """LangChain run config that traces one chat request, or an empty dict when off.

        The session ID is hashed so raw client identifiers never reach the trace store.

        Args:
            request_id: Request ID, stored as trace metadata.
            session_id: Optional client session ID.

        Returns:
            Callbacks, run name and metadata to merge into the graph invocation config.
        """
        if self.client is None:
            return {}
        try:
            handler = langfuse_langchain.CallbackHandler(public_key=self._public_key)
        except Exception as exc:
            logger.warning("tracing_handler_failed", extra={"error": type(exc).__name__})
            return {}
        session = hashlib.sha256((session_id or request_id).encode()).hexdigest()[:16]
        return {
            "callbacks": [handler],
            "run_name": "chat",
            "metadata": {
                "langfuse_session_id": session,
                "langfuse_tags": ["aeroassist"],
                "request_id": request_id,
            },
        }

    def flush(self) -> None:
        """Sends buffered spans; called on shutdown."""
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception as exc:
            logger.warning("tracing_flush_failed", extra={"error": type(exc).__name__})
