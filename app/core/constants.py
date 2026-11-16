"""Domain constants shared across modules: tool names, identifier formats, dates and places."""

from __future__ import annotations

import re

FLIGHT_STATUS_TOOL = "get_flight_status"
BOOKING_CONTEXT_TOOL = "get_booking_context"
SEARCH_TOOL = "search_reaccommodation"
WRITE_TOOL = "create_reaccommodation_hold"
READ_TOOL_NAMES = frozenset({FLIGHT_STATUS_TOOL, BOOKING_CONTEXT_TOOL, SEARCH_TOOL})
TOOL_NAMES = READ_TOOL_NAMES | {WRITE_TOOL}

FLIGHT_PATTERN = re.compile(r"\b[A-Z]{2}\d{1,4}\b", re.IGNORECASE)
BOOKING_PATTERN = re.compile(
    r"\b(?![A-Z]{2}\d{1,4}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b", re.IGNORECASE
)
ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

RELATIVE_DAYS = {"pasado mañana": 2, "mañana": 1, "hoy": 0, "ayer": -1}
HOLD_WINDOW_DAYS = 3
CITY_TO_IATA = {"santiago": "SCL", "lima": "LIM", "miami": "MIA", "los ángeles": "LAX"}

MODEL_PRICES_USD_PER_MILLION = {
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "gpt-6.1-terra": {"input": 1.00, "cached_input": 0.05, "output": 5.00},
    "gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
    "gpt-6-astra": {"input": 10.00, "cached_input": 1.00, "output": 50.00},
}
