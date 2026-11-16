"""PII redaction shared by the JSON logger and trace exporters.

Booking codes and personal identifiers are replaced by type tags. Flight numbers, airport
codes and dates are operational data and are preserved.
"""

from __future__ import annotations

import re

import pydantic

from app.core import constants

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
_RUT = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b")
_PHONE = re.compile(r"\+\d{1,3}(?:[\s.-]?\d){7,12}\b|\b9(?:\s?\d){8}\b")
_DNI = re.compile(r"\b\d{8}\b")
_PASSPORT = re.compile(r"\b[A-Z]{1,2}\d{6,8}\b", re.IGNORECASE)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _redact_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return "[CARD]" if _luhn_valid(digits) else match.group()


def redact_personal(text: str) -> str:
    """Replaces personal data but keeps operational identifiers; used on text sent to the LLM.

    Booking codes and flight numbers stay, because the model needs them to call tools.

    Args:
        text: User message or other free text.

    Returns:
        The text with emails, URLs, IPs, cards, RUT, phones, DNI and passports replaced by tags.
    """
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    text = _IP.sub("[IP]", text)
    text = _CARD.sub(_redact_card, text)
    text = _RUT.sub("[RUT]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _DNI.sub("[DNI]", text)
    return _PASSPORT.sub("[PASSPORT]", text)


def redact(text: str) -> str:
    """Replaces PII in free text with type tags; used for logs and traces.

    Args:
        text: Any string headed for logs or traces.

    Returns:
        The text with personal data and booking codes replaced by tags such as [EMAIL] or
        [BOOKING].
    """
    return constants.BOOKING_PATTERN.sub("[BOOKING]", redact_personal(text))


def redact_value(value: object) -> object:
    """Redacts every string inside a nested value, failing closed on unknown objects.

    Dicts, lists and tuples keep their structure, pydantic models are dumped first, numbers,
    booleans and None pass through, and any other object is converted to text and redacted, so
    nothing reaches a trace unmasked.

    Args:
        value: Any value, such as a trace input or output.

    Returns:
        A JSON-serializable copy with every string redacted.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, pydantic.BaseModel):
        return redact_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): redact_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [redact_value(item) for item in value]
    return redact(str(value))
