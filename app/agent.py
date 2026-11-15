"""Application orchestration boundary for the progressive levels."""

from __future__ import annotations

import re
import time

from app import llm, schemas

_HELP = re.compile(
    r"qu[eé] (puedes|sabes) hacer|qu[eé] (tipo de )?ayuda|c[oó]mo (me )?puedes ayudar",
    re.IGNORECASE,
)

HELP_ANSWER = (
    "Puedo ayudarte con: el estado de un vuelo (número y fecha), el contexto mínimo de una "
    "reserva, alternativas de reacomodación para una ruta y fecha, y las políticas de "
    "reacomodación, asistencia, equipaje, reembolso y asistencia especial. Para crear un hold "
    "de reacomodación necesito tu confirmación explícita."
)

UNSUPPORTED_ANSWER = (
    "No puedo responder esa consulta: está fuera de mi alcance o no tengo evidencia suficiente. "
    "Puedo ayudarte con vuelos, reservas, alternativas de reacomodación y políticas de la "
    "aerolínea."
)


class AeroAssistAgent:
    """Answers chat requests.

    This revision only routes the help intent and abstains on everything else; the LLM
    classifier, MCP tools and RAG replace the keyword route in later stages.

    Args:
        provider: LLM backend whose model name is reported.
    """

    def __init__(self, provider: llm.LLMProvider) -> None:
        self._llm = provider

    async def chat(self, request: schemas.ChatRequest, *, request_id: str) -> schemas.ChatResponse:
        """Builds the response for one chat request.

        Args:
            request: Validated chat request.
            request_id: ID echoed in the response.

        Returns:
            The structured chat response.
        """
        started = time.perf_counter()
        is_help = bool(_HELP.search(request.message))
        return schemas.ChatResponse(
            answer=HELP_ANSWER if is_help else UNSUPPORTED_ANSWER,
            request_id=request_id,
            model=self._llm.model_name,
            latency_ms=round((time.perf_counter() - started) * 1000),
            abstained=not is_help,
            route="help" if is_help else "unsupported",
        )
