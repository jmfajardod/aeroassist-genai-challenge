"""Argument provenance: identifiers and dates must come from the user or earlier tool results."""

from __future__ import annotations

import datetime
import json
import re
from typing import Any

import pydantic

from app.core import constants


def resolve_dates(message: str, reference_date: datetime.date) -> set[str]:
    """ISO dates the user stated, plus relative dates resolved against the reference date.

    Args:
        message: User message.
        reference_date: Date that "hoy" refers to.

    Returns:
        Dates in YYYY-MM-DD form.
    """
    dates = set(constants.ISO_DATE_PATTERN.findall(message))
    remaining = message.lower()
    for word, offset in constants.RELATIVE_DAYS.items():
        if word in remaining:
            dates.add((reference_date + datetime.timedelta(days=offset)).isoformat())
            remaining = remaining.replace(word, " ")
    return dates


class Provenance(pydantic.BaseModel):
    """Values an argument may take: stated by the user or returned by an earlier tool call."""

    text: str
    dates: set[str]

    def add_result(self, result: dict[str, Any]) -> None:
        """Makes identifiers and dates from a tool result usable in later calls."""
        dumped = json.dumps(result)
        self.text += " " + dumped
        self.dates |= set(constants.ISO_DATE_PATTERN.findall(dumped))

    def missing(self, args: pydantic.BaseModel) -> list[str]:
        """Names of identifier or date arguments that the user never provided.

        Origin and destination are exempt because city names map to IATA codes.

        Args:
            args: Validated tool arguments.

        Returns:
            The argument names without provenance; empty when every value is allowed.
        """
        upper = self.text.upper()
        missing = []
        for field in ("flight_number", "booking_code"):
            value = getattr(args, field, None)
            if value is not None and not re.search(rf"\b{value}\b", upper):
                missing.append(field)
        date = getattr(args, "departure_date", None)
        if date is not None and date.isoformat() not in self.dates:
            missing.append("departure_date")
        return missing
