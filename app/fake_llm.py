"""Deterministic, offline LLM backend used by tests, CI and keyless local runs.

It imitates the real provider with keyword rules: it classifies messages and builds templated
answers from the evidence it is given. It never calls the network.
"""

from __future__ import annotations

import re
from typing import TypeVar

import pydantic

from app import llm_schemas, prompting

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
_WRITE = re.compile(r"\bhold\b|reserva(r)? el asiento|ret[eé]n", re.IGNORECASE)
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


def classify(message: str) -> llm_schemas.Classification:
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
    return llm_schemas.Classification(
        route=route,
        needs_tools=needs_tools,
        needs_policy=needs_policy,
        write_requested=bool(_WRITE.search(message)),
    )


def answer(variables: dict[str, str]) -> llm_schemas.Answer:
    """Templated answer that cites every chunk it receives.

    Args:
        variables: The variables rendered into the answer prompt.

    Returns:
        A grounded answer when evidence exists, otherwise an ungrounded one.
    """
    chunk_ids = re.findall(r"chunk_id: (\S+)", variables.get("chunks", ""))
    has_tools = variables.get("tool_results", "").strip() not in ("", "(ninguno)")
    if not chunk_ids and not has_tools:
        return llm_schemas.Answer(
            answer="No puedo responder con la evidencia disponible.",
            cited_chunk_ids=[],
            grounding="ungrounded",
        )
    return llm_schemas.Answer(
        answer="Respuesta basada en la evidencia disponible: "
        + variables.get("chunks", "")
        + variables.get("tool_results", ""),
        cited_chunk_ids=chunk_ids,
        grounding="grounded",
    )


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
        result = classify(variables["message"]) if schema is llm_schemas.Classification else None
        if schema is llm_schemas.Answer:
            result = answer(variables)
        assert isinstance(result, schema)
        return result
