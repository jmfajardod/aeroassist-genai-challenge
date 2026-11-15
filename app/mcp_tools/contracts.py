"""Argument and result contracts of the MCP tools, plus the function specs shown to the model.

Argument models are stricter than the server's own signatures; result models validate the
untrusted tool output.
"""

from __future__ import annotations

import datetime
from typing import Any

import pydantic
from langchain_core.utils import function_calling

from app.core import constants


def _upper(value: object) -> object:
    return value.strip().upper() if isinstance(value, str) else value


class _Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class FlightStatusArgs(_Args):
    """Current status, delay and disruption cause of one flight on one date."""

    flight_number: str = pydantic.Field(pattern=r"^[A-Z]{2}\d{1,4}$", description="e.g. LA401")
    departure_date: datetime.date = pydantic.Field(description="YYYY-MM-DD")

    _normalize = pydantic.field_validator("flight_number", mode="before")(_upper)


class BookingContextArgs(_Args):
    """Minimum context of one booking: route, date, fare, loyalty tier and special assistance."""

    booking_code: str = pydantic.Field(pattern=r"^[A-Z0-9]{6}$", description="6 characters")

    _normalize = pydantic.field_validator("booking_code", mode="before")(_upper)


class SearchArgs(_Args):
    """Rebooking alternatives with free seats for one route and date."""

    origin: str = pydantic.Field(pattern=r"^[A-Z]{3}$", description="IATA code, e.g. SCL")
    destination: str = pydantic.Field(pattern=r"^[A-Z]{3}$", description="IATA code, e.g. LIM")
    departure_date: datetime.date = pydantic.Field(description="YYYY-MM-DD")
    max_results: int = pydantic.Field(default=3, description="1 to 5")

    _normalize = pydantic.field_validator("origin", "destination", mode="before")(_upper)

    @pydantic.field_validator("max_results")
    @classmethod
    def clamp(cls, value: int) -> int:
        """Clamps the result count to the server's 1..5 range."""
        return max(1, min(value, 5))


class HoldArgs(_Args):
    """Arguments of the write tool; only the write gate builds these."""

    booking_code: str = pydantic.Field(pattern=r"^[A-Z0-9]{6}$")
    flight_number: str = pydantic.Field(pattern=r"^[A-Z]{2}\d{1,4}$")
    idempotency_key: str = pydantic.Field(pattern=r"^[A-Za-z0-9._:-]{1,64}$")


READ_TOOLS: dict[str, type[_Args]] = {
    constants.FLIGHT_STATUS_TOOL: FlightStatusArgs,
    constants.BOOKING_CONTEXT_TOOL: BookingContextArgs,
    constants.SEARCH_TOOL: SearchArgs,
}


def read_tool_specs() -> list[dict[str, Any]]:
    """OpenAI-style function specs for the read tools offered to the model."""
    specs = []
    for name, model in READ_TOOLS.items():
        spec = function_calling.convert_to_openai_tool(model)
        spec["function"]["name"] = name
        specs.append(spec)
    return specs


class ToolResult(pydantic.BaseModel):
    """Base for validated tool results; unknown fields are ignored."""

    model_config = pydantic.ConfigDict(extra="ignore", frozen=True)


class BookingContext(ToolResult):
    """Result of get_booking_context; fields other than found are absent when not found."""

    found: bool = True
    booking_code: str
    flight_number: str | None = None
    departure_date: datetime.date | None = None
    origin: str | None = None
    destination: str | None = None
    special_service_requests: list[str] = []


class Alternative(ToolResult):
    """One rebooking option returned by search_reaccommodation."""

    flight_number: str
    departure_date: datetime.date
    departure_time: str
    origin: str
    destination: str
    available_seats: int


class SearchResult(ToolResult):
    """Result of search_reaccommodation."""

    options: list[Alternative]


class HoldResult(ToolResult):
    """Result of create_reaccommodation_hold."""

    created: bool
    hold_id: str | None = None
    reason: str | None = None
    expires_in_minutes: int | None = None


class ToolOutput(pydantic.BaseModel):
    """An executed read call's raw result, passed to the answer model as evidence."""

    model_config = pydantic.ConfigDict(frozen=True)

    tool: str
    result: dict[str, Any]
