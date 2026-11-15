"""Structured JSON logging with every string field redacted."""

from __future__ import annotations

import datetime
import json
import logging
import sys

from app.observability import redaction

logger = logging.getLogger("aeroassist")

_RESERVED_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Formats records as one JSON object per line with every string field redacted."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializes the record, including extra fields, as redacted JSON.

        Args:
            record: The log record.

        Returns:
            A single-line JSON string using Cloud Logging's severity key.
        """
        payload: dict[str, object] = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, datetime.UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": redaction.redact(record.getMessage()),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = redaction.redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = redaction.redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    """Routes the aeroassist logger to stdout as JSON.

    Args:
        level: Minimum level name, e.g. "INFO".
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
