"""PII redaction shared by the JSON logger and trace exporters.

Booking codes and personal identifiers are replaced by type tags. Flight numbers, airport
codes and dates are operational data and are preserved.
"""

from __future__ import annotations

import re

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


def redact(text: str) -> str:
    """Replaces PII in free text with type tags.

    Args:
        text: Any string headed for logs or traces.

    Returns:
        The text with emails, URLs, IPs, cards, RUT, phones, DNI, passports and booking
        codes replaced by tags such as [EMAIL] or [BOOKING].
    """
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    text = _IP.sub("[IP]", text)
    text = _CARD.sub(_redact_card, text)
    text = _RUT.sub("[RUT]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _DNI.sub("[DNI]", text)
    text = _PASSPORT.sub("[PASSPORT]", text)
    return constants.BOOKING_PATTERN.sub("[BOOKING]", text)
