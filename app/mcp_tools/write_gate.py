"""Deterministic hold flow: the only code path that can call the write tool.

The model never authorizes a write. The booking, the target flight and the idempotency key
come from the current message, the booking and the target are checked through read tools, and
the hold is created only when confirm_action is true for this same request.
"""

from __future__ import annotations

import datetime
import hashlib
import re

import pydantic

from app import schemas
from app.core import constants, errors
from app.mcp_tools import contracts, runner

_KEY = re.compile(
    r"(?:idempotency[\s_-]*key|clave(?:\s+de\s+idempotencia)?|key)\s*[:=]?\s*([A-Za-z0-9._:-]{1,64})",
    re.IGNORECASE,
)


class WriteResult(pydantic.BaseModel):
    """Outcome of the hold flow."""

    model_config = pydantic.ConfigDict(frozen=True)

    answer: str
    requires_confirmation: bool
    tool_calls: list[schemas.ToolCallTrace]


def derive_idempotency_key(booking_code: str, flight_number: str, departure_date: str) -> str:
    """Stable key used when the user gives none, so retries of one request reuse one hold."""
    digest = hashlib.sha256(f"{booking_code}|{flight_number}|{departure_date}".encode())
    return f"auto-{digest.hexdigest()[:16]}"


def _ssr_note(booking: contracts.BookingContext) -> str:
    ssr = booking.special_service_requests
    if not ssr:
        return ""
    return (
        f" La reserva tiene asistencia especial {', '.join(ssr)}: se debe preservar y la "
        "confirmación final depende del vuelo alternativo."
    )


def _window(base: datetime.date) -> list[datetime.date]:
    window = constants.HOLD_WINDOW_DAYS
    offsets = sorted(range(-window, window + 1), key=abs)
    return [base + datetime.timedelta(days=offset) for offset in offsets]


async def handle(message: str, confirm_action: bool, tool_runner: runner.ToolRunner) -> WriteResult:
    """Validates a hold request and creates the hold only if it is confirmed and valid.

    Args:
        message: User message, the only source of the booking, flight and key.
        confirm_action: Confirmation flag of this request.
        tool_runner: Tool runner used for the read checks and the write.

    Returns:
        The answer, whether confirmation is still required, and the executed tool calls.

    Raises:
        errors.UnsafeRequestError: If a confirmed request does not name exactly one booking and
            at least one flight.
    """
    bookings = sorted({code.upper() for code in constants.BOOKING_PATTERN.findall(message)})
    flights = sorted({code.upper() for code in constants.FLIGHT_PATTERN.findall(message)})
    if len(bookings) != 1 or not flights:
        if confirm_action:
            raise errors.UnsafeRequestError("A hold needs exactly one booking code and a flight.")
        answer = "Para preparar un hold necesito un código de reserva y el vuelo alternativo."
        return WriteResult(answer=answer, requires_confirmation=False, tool_calls=[])

    traces: list[schemas.ToolCallTrace] = []
    trace, booking = await tool_runner.execute_typed(
        constants.BOOKING_CONTEXT_TOOL,
        contracts.BookingContextArgs(booking_code=bookings[0]),
        contracts.BookingContext,
    )
    traces.append(trace)
    if not booking.found or booking.departure_date is None:
        answer = (
            f"No encontré la reserva {bookings[0]}. Verifica el código. "
            "No se ha creado ningún hold."
        )
        return WriteResult(answer=answer, requires_confirmation=False, tool_calls=traces)

    targets = [flight for flight in flights if flight != booking.flight_number]
    if len(targets) != 1:
        answer = (
            "Indica un único vuelo alternativo para el hold; con varios vuelos no puedo pedir "
            "una confirmación inequívoca. No se ha creado ningún hold."
        )
        return WriteResult(answer=answer, requires_confirmation=len(targets) > 1, tool_calls=traces)
    target = targets[0]

    option = None
    seen: list[str] = []
    for date in _window(booking.departure_date):
        trace, found = await tool_runner.execute_typed(
            constants.SEARCH_TOOL,
            contracts.SearchArgs(
                origin=booking.origin,
                destination=booking.destination,
                departure_date=date,
                max_results=5,
            ),
            contracts.SearchResult,
        )
        traces.append(trace)
        option = next((o for o in found.options if o.flight_number == target), None)
        if option:
            break
        seen += [
            f"{o.flight_number} ({o.departure_date} {o.departure_time})" for o in found.options
        ]
    route = f"{booking.origin}→{booking.destination}"
    if option is None:
        answer = (
            f"El vuelo {target} no es una alternativa válida para la reserva {bookings[0]}: no "
            f"tiene cupos en su ruta {route} dentro de ±{constants.HOLD_WINDOW_DAYS} días de la "
            "fecha original. "
            "No se ha creado ningún hold."
        )
        if seen:
            answer += f" Alternativas válidas: {', '.join(seen[:3])}."
        return WriteResult(answer=answer, requires_confirmation=False, tool_calls=traces)

    summary = (
        f"la reserva {bookings[0]} en el vuelo {target} ({route}, {option.departure_date} "
        f"{option.departure_time})"
    )
    if not confirm_action:
        answer = (
            f"Puedo crear un hold para {summary}.{_ssr_note(booking)} Para continuar necesito tu "
            "confirmación explícita. No se ha creado ningún hold."
        )
        return WriteResult(answer=answer, requires_confirmation=True, tool_calls=traces)

    match = _KEY.search(message)
    key = (
        match.group(1).rstrip(".:")
        if match
        else derive_idempotency_key(bookings[0], target, option.departure_date.isoformat())
    )
    trace, hold = await tool_runner.execute_typed(
        constants.WRITE_TOOL,
        contracts.HoldArgs(booking_code=bookings[0], flight_number=target, idempotency_key=key),
        contracts.HoldResult,
    )
    traces.append(trace)
    if not hold.created:
        reason = hold.reason or "sin motivo"
        answer = f"No se pudo crear el hold ({reason}). Ningún hold quedó activo."
        return WriteResult(answer=answer, requires_confirmation=False, tool_calls=traces)
    answer = (
        f"Hold {hold.hold_id} creado para {summary}; expira en {hold.expires_in_minutes} "
        f"minutos.{_ssr_note(booking)}"
    )
    return WriteResult(answer=answer, requires_confirmation=False, tool_calls=traces)
