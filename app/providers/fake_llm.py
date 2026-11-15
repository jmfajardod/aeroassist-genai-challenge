"""Deterministic, offline LLM backend used by tests, CI and keyless local runs.

It imitates the real provider with keyword rules: it classifies messages and builds templated
answers from the evidence it is given. It never calls the network.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, TypeVar

import pydantic

from app.core import constants
from app.mcp_tools import provenance
from app.orchestration import prompting
from app.providers import outputs

SchemaT = TypeVar("SchemaT", bound=pydantic.BaseModel)

_INJECTION = re.compile(
    r"ignor[ae]\w*\s+(\w+\s+){0,3}(instruc|instru[cç][õo]es|previous|anteriores)"
    r"|system prompt|prompt de sistema|variables de entorno|environment variables|api keys?"
    r"|^\s*system\s*:|\bsystem\s*:|sin confirmaci[oó]n|sem confirma[cç][aã]o",
    re.IGNORECASE,
)
_HELP = re.compile(
    r"qu[eé] (puedes|sabes) hacer|qu[eé] (tipo de )?ayuda|en qu[eé] me puedes ayudar"
    r"|c[oó]mo (me )?puedes ayudar",
    re.IGNORECASE,
)
_WRITE = re.compile(r"\bholds?\b|reserva(r)? el asiento|ret[eé]n", re.IGNORECASE)
_TOOLS = re.compile(
    r"\b[A-Z]{2}\d{1,4}\b|\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b"
    r"|alternativ|opciones|estado del vuelo|reacomodar",
    re.IGNORECASE,
)
_POLICY = re.compile(
    r"equipaje|maleta|asistencia|hotel|comida|refrigerio|reembolso|devoluci|tarifa|penalidad"
    r"|silla de ruedas|wchr|wchs|wchc|categor|gold|platinum|prioridad|confirmaci|credencial"
    r"|reacomodaci|cancelaci|diagn[oó]stico|espera",
    re.IGNORECASE,
)


def classify(message: str) -> outputs.Classification:
    """Rule-based stand-in for the LLM classifier.

    Args:
        message: User message.

    Returns:
        The routing decision.
    """
    needs_tools = bool(_TOOLS.search(message))
    needs_policy = bool(_POLICY.search(message))
    if _INJECTION.search(message):
        route = "injection"
    elif _HELP.search(message):
        route = "help"
    elif needs_tools or needs_policy or _WRITE.search(message):
        route = "answer"
    else:
        route = "unsupported"
    return outputs.Classification(
        route=route,
        needs_tools=needs_tools,
        needs_policy=needs_policy,
        write_requested=bool(_WRITE.search(message)),
    )


def answer(variables: dict[str, str]) -> outputs.Answer:
    """Templated answer that cites every chunk it receives.

    Args:
        variables: The variables rendered into the answer prompt.

    Returns:
        A grounded answer when evidence exists, otherwise an ungrounded one.
    """
    chunk_ids = re.findall(r"chunk_id: (\S+)", variables.get("chunks", ""))
    has_tools = variables.get("tool_results", "").strip() not in ("", "(ninguno)")
    if not chunk_ids and not has_tools:
        return outputs.Answer(
            answer="No puedo responder con la evidencia disponible.",
            cited_chunk_ids=[],
            grounding="ungrounded",
        )
    return outputs.Answer(
        answer="Respuesta basada en la evidencia disponible: "
        + variables.get("chunks", "")
        + variables.get("tool_results", ""),
        cited_chunk_ids=chunk_ids,
        grounding="grounded",
    )


_ROUTE = re.compile(r"\b([A-Z]{3})\s+(?:a|->|→)\s+([A-Z]{3})\b")
_MAX = re.compile(r"hasta (\d+)")
_WANTS_OPTIONS = re.compile(r"alternativ|opciones|reacomodar", re.IGNORECASE)
_WANTS_STATUS = re.compile(r"estado|c[oó]mo (est[aá]|viene)|a tiempo|sale", re.IGNORECASE)


def _route(message: str) -> tuple[str, str] | None:
    match = _ROUTE.search(message)
    if match:
        return match.group(1), match.group(2)
    found = [(message.lower().find(name), code) for name, code in constants.CITY_TO_IATA.items()]
    codes = [code for position, code in sorted(found) if position >= 0]
    return (codes[0], codes[1]) if len(codes) >= 2 else None


def first_tool_calls(message: str, reference_date: datetime.date) -> list[dict[str, Any]]:
    """Rule-based first round of read tool calls for a message."""
    dates = sorted(provenance.resolve_dates(message, reference_date))
    flights = [f.upper() for f in constants.FLIGHT_PATTERN.findall(message)]
    bookings = [b.upper() for b in constants.BOOKING_PATTERN.findall(message)]
    calls: list[dict[str, Any]] = []
    if bookings:
        calls.append(
            {"name": constants.BOOKING_CONTEXT_TOOL, "args": {"booking_code": bookings[0]}}
        )
    if flights and dates:
        calls.append(
            {
                "name": constants.FLIGHT_STATUS_TOOL,
                "args": {"flight_number": flights[0], "departure_date": dates[0]},
            }
        )
    route = _route(message)
    if route and dates:
        args: dict[str, Any] = {
            "origin": route[0],
            "destination": route[1],
            "departure_date": dates[0],
        }
        if match := _MAX.search(message):
            args["max_results"] = int(match.group(1))
        calls.append({"name": constants.SEARCH_TOOL, "args": args})
    return calls


def follow_up_calls(message: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Second-round calls that use a booking context from the first round."""
    booking = next((r for r in results if r.get("booking_code") and r.get("found")), None)
    if booking is None:
        return []
    if _WANTS_OPTIONS.search(message):
        args = {
            "origin": booking["origin"],
            "destination": booking["destination"],
            "departure_date": booking["departure_date"],
        }
        return [{"name": constants.SEARCH_TOOL, "args": args}]
    if _WANTS_STATUS.search(message):
        args = {
            "flight_number": booking["flight_number"],
            "departure_date": booking["departure_date"],
        }
        return [{"name": constants.FLIGHT_STATUS_TOOL, "args": args}]
    return []


class FakeLLM:
    """Offline provider that dispatches on the requested output schema."""

    @property
    def model_name(self) -> str:
        """Identifier reported in responses."""
        return "fake-deterministic"

    async def structured(
        self, *, role: str, prompt: prompting.RenderedPrompt, schema: type[SchemaT]
    ) -> SchemaT:
        """Returns the rule-based output for the schema.

        Args:
            role: Ignored; kept for interface parity.
            prompt: Rendered prompt; only its variables are read.
            schema: Classification or Answer.

        Returns:
            An instance of schema.
        """
        del role
        variables = dict(prompt.variables)
        result = classify(variables["message"]) if schema is outputs.Classification else None
        if schema is outputs.Answer:
            result = answer(variables)
        assert isinstance(result, schema)
        return result

    async def choose_tools(
        self,
        *,
        prompt: prompting.RenderedPrompt,
        rounds: list[outputs.ToolRound],
        tools: list[dict[str, Any]],
    ) -> outputs.ToolStep:
        """Proposes read calls with the same rules on every run.

        Args:
            prompt: Rendered tools prompt; reads the message and reference date variables.
            rounds: Earlier calls and results.
            tools: Ignored; the fake knows the read tools.

        Returns:
            The proposed calls, or a request for the missing data.
        """
        del tools
        message = prompt.variables["message"]
        if not rounds:
            reference = datetime.date.fromisoformat(prompt.variables["reference_date"])
            calls = first_tool_calls(message, reference)
        elif len(rounds) == 1:
            calls = follow_up_calls(message, [result for _, result in rounds[0]])
        else:
            calls = []
        if not calls and not rounds:
            return outputs.ToolStep(
                requests=[],
                text="Indica el número de vuelo y fecha, el código de reserva o la ruta y fecha.",
            )
        requests = [
            outputs.ToolRequest(id=f"fake-{len(rounds)}-{i}", name=c["name"], arguments=c["args"])
            for i, c in enumerate(calls)
        ]
        return outputs.ToolStep(requests=requests)
