"""PII redaction for logs and traces."""

from __future__ import annotations

import io
import json
import logging

import pytest

from app.observability import log, redaction


@pytest.mark.parametrize(
    ("text", "secret", "tag"),
    [
        ("reserva Q7M2PX por favor", "Q7M2PX", "[BOOKING]"),
        ("reserva z3trip", "z3trip", "[BOOKING]"),
        ("escribir a ana.perez@example.com", "ana.perez@example.com", "[EMAIL]"),
        ("llamar al +56 9 8765 4321", "8765 4321", "[PHONE]"),
        ("celular 987654321", "987654321", "[PHONE]"),
        ("RUT 12.345.678-5", "12.345.678-5", "[RUT]"),
        ("DNI 45678912", "45678912", "[DNI]"),
        ("pasaporte P1234567", "P1234567", "[PASSPORT]"),
        ("tarjeta 4111 1111 1111 1111", "4111 1111 1111 1111", "[CARD]"),
        ("desde 10.0.0.12", "10.0.0.12", "[IP]"),
        ("ver https://example.com/r?code=Q7M2PX", "example.com", "[URL]"),
    ],
)
def test_redacts_pii(text: str, secret: str, tag: str) -> None:
    redacted = redaction.redact(text)
    assert secret not in redacted
    assert tag in redacted


def test_preserves_operational_identifiers() -> None:
    text = "LA401 y LA2376 SCL→LIM el 2026-08-06, hold HOLD-1a2b3c4d5e6f"
    assert redaction.redact(text) == text


def test_json_log_redacts_message_and_extra_fields() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(log.JsonFormatter())
    logger = logging.getLogger("test_redaction")
    logger.addHandler(handler)
    logger.propagate = False
    logger.warning("booking Q7M2PX", extra={"detail": "ana@example.com", "status": 200})
    record = json.loads(stream.getvalue())
    assert record["message"] == "booking [BOOKING]"
    assert record["detail"] == "[EMAIL]"
    assert record["status"] == 200
