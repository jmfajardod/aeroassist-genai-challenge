"""LangGraph flow: classify, fan out to evidence branches, answer, then validate.

Template routes (help, unsupported, injection, write without tools) never call the answer
model. Evidence branches run in parallel through Send and join at the answer node.
"""

from __future__ import annotations

import dataclasses
import json
import operator
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from langgraph import graph as lg_graph
from langgraph import types as lg_types

from app import config, llm, llm_schemas, prompting, rag, schemas

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
WRITE_CONFIRMATION_ANSWER = (
    "Para crear un hold de reacomodación necesito tu confirmación explícita de la reserva y del "
    "vuelo alternativo. No se ha creado ningún hold."
)
WRITE_UNAVAILABLE_ANSWER = (
    "No puedo crear holds de reacomodación en esta versión del asistente. No se ha creado "
    "ningún hold."
)
NO_EVIDENCE_ANSWER = (
    "No tengo evidencia suficiente para responder esa consulta. Indica el dato que falta o "
    "escala el caso a un agente."
)


@dataclasses.dataclass(frozen=True)
class Outcome:
    """Final, validated result of one graph run."""

    route: str
    answer: str
    citations: list[schemas.Citation] = dataclasses.field(default_factory=list)
    tool_calls: list[schemas.ToolCallTrace] = dataclasses.field(default_factory=list)
    abstained: bool = False
    grounding: str = "not_applicable"
    requires_confirmation: bool = False
    similarity: schemas.Similarity | None = None


def _merge(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {**left, **right}


class State(TypedDict, total=False):
    """Graph state; list and timing fields are merged across parallel branches."""

    message: str
    confirm_action: bool
    classification: llm_schemas.Classification
    chunks: Annotated[list[rag.RetrievedChunk], operator.add]
    tool_calls: Annotated[list[schemas.ToolCallTrace], operator.add]
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    draft: llm_schemas.Answer
    outcome: Outcome
    timings_ms: Annotated[dict[str, int], _merge]


Node = Callable[[State], Awaitable[dict[str, Any]]]


def _timed(name: str, node: Node) -> Node:
    async def run(state: State) -> dict[str, Any]:
        started = time.perf_counter()
        update = await node(state)
        update["timings_ms"] = {name: round((time.perf_counter() - started) * 1000)}
        return update

    return run


async def classify(
    provider: llm.LLMProvider, message: str, confirm_action: bool
) -> llm_schemas.Classification:
    """Asks the classifier model to route one message.

    Args:
        provider: LLM backend.
        message: User message.
        confirm_action: The request's confirmation flag, so a plain confirmation is not
            mistaken for a forged one.

    Returns:
        The routing decision.
    """
    prompt = prompting.load("classifier").render(
        message=message, confirm_action=str(confirm_action).lower()
    )
    return await provider.structured(
        role="classifier", prompt=prompt, schema=llm_schemas.Classification
    )


def _format_chunks(chunks: list[rag.RetrievedChunk]) -> str:
    if not chunks:
        return "(ninguno)"
    return "\n\n".join(
        f"[chunk_id: {c.chunk_id} | doc_id: {c.doc_id} | title: {c.title}]\n{c.text}"
        for c in chunks
    )


def _format_tool_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(ninguno)"
    return "\n".join(json.dumps(result, ensure_ascii=False) for result in results)


def _template(state: State) -> Outcome:
    decision = state["classification"]
    if decision.route == "injection":
        return Outcome(route="injection", answer="")
    if decision.route == "help":
        return Outcome(route="help", answer=HELP_ANSWER)
    if decision.route == "unsupported":
        return Outcome(
            route="unsupported", answer=UNSUPPORTED_ANSWER, abstained=True, grounding="ungrounded"
        )
    if not state.get("confirm_action"):
        return Outcome(route="write", answer=WRITE_CONFIRMATION_ANSWER, requires_confirmation=True)
    return Outcome(route="write", answer=WRITE_UNAVAILABLE_ANSWER, abstained=True)


def _evidence_route(decision: llm_schemas.Classification) -> str:
    parts = [
        name
        for name, wanted in (("tools", decision.needs_tools), ("policy", decision.needs_policy))
        if wanted
    ]
    return "+".join(parts) or "none"


def validate(state: State) -> Outcome:
    """Keeps only citations that were retrieved and cited, and applies abstention.

    Args:
        state: State after the answer node.

    Returns:
        The final outcome.
    """
    draft = state["draft"]
    chunks = state.get("chunks", [])
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    cited = [by_id[cid] for cid in dict.fromkeys(draft.cited_chunk_ids) if cid in by_id]
    abstained = draft.grounding == "ungrounded"
    citations = (
        []
        if abstained
        else [
            schemas.Citation(doc_id=c.doc_id, title=c.title, chunk_id=c.chunk_id, score=c.score)
            for c in cited
        ]
    )
    scores = [chunk.score for chunk in chunks]
    similarity = (
        schemas.Similarity(top=max(scores), mean=sum(scores) / len(scores)) if scores else None
    )
    return Outcome(
        route=_evidence_route(state["classification"]),
        answer=draft.answer,
        citations=citations,
        tool_calls=state.get("tool_calls", []),
        abstained=abstained,
        grounding=draft.grounding,
        similarity=similarity,
    )


def build(
    provider: llm.LLMProvider,
    settings: config.Settings,
    retriever: rag.Retriever | None = None,
    top_k: int = 3,
) -> Any:
    """Compiles the answer graph.

    Args:
        provider: LLM backend for the classifier and answer roles.
        settings: Application settings.
        retriever: Policy retriever; without one the policy branch is skipped.
        top_k: Chunks requested from the retriever.

    Returns:
        A compiled LangGraph runnable that takes and returns State.
    """
    reference_date = config.load_variant(settings.variant_path).reference_date.isoformat()

    async def classify_node(state: State) -> dict[str, Any]:
        decision = await classify(provider, state["message"], state.get("confirm_action", False))
        return {"classification": decision}

    def after_classify(state: State) -> list[lg_types.Send] | str:
        decision = state["classification"]
        if decision.route != "answer" or decision.write_requested:
            return "respond"
        sends = []
        if decision.needs_policy and retriever is not None:
            sends.append(lg_types.Send("retrieve", state))
        return sends or "answer"

    async def respond_node(state: State) -> dict[str, Any]:
        return {"outcome": _template(state)}

    async def retrieve_node(state: State) -> dict[str, Any]:
        assert retriever is not None
        return {"chunks": await retriever.search(state["message"], top_k=top_k)}

    async def answer_node(state: State) -> dict[str, Any]:
        chunks = state.get("chunks", [])
        results = state.get("tool_results", [])
        if not chunks and not results:
            draft = llm_schemas.Answer(
                answer=NO_EVIDENCE_ANSWER, cited_chunk_ids=[], grounding="ungrounded"
            )
            return {"draft": draft}
        prompt = prompting.load("answer").render(
            reference_date=reference_date,
            chunks=_format_chunks(chunks),
            tool_results=_format_tool_results(results),
            message=state["message"],
        )
        draft = await provider.structured(role="answer", prompt=prompt, schema=llm_schemas.Answer)
        return {"draft": draft}

    async def validate_node(state: State) -> dict[str, Any]:
        return {"outcome": validate(state)}

    builder = lg_graph.StateGraph(State)
    builder.add_node("classify", _timed("classify", classify_node))
    builder.add_node("respond", _timed("respond", respond_node))
    builder.add_node("retrieve", _timed("retrieve", retrieve_node))
    builder.add_node("answer", _timed("answer", answer_node))
    builder.add_node("validate", _timed("validate", validate_node))
    builder.add_edge(lg_graph.START, "classify")
    builder.add_conditional_edges("classify", after_classify, ["respond", "retrieve", "answer"])
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "validate")
    builder.add_edge("validate", lg_graph.END)
    builder.add_edge("respond", lg_graph.END)
    return builder.compile()
