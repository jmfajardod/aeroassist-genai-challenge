"""LangGraph flow: classify, fan out to evidence branches, answer, then validate.

Template routes (help, unsupported, injection, write without tools) never call the answer
model. Evidence branches run in parallel through Send and join at the answer node. Retrieval
runs on every answer route because it is cheap and the classifier's policy flag misses some
policy questions; the evidence gate, the answer prompt and citation validation keep it from
adding noise. When the tool step asks for missing data, the answer model receives that request
together with any policy evidence and decides what to use.
"""

from __future__ import annotations

import datetime
import operator
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import pydantic
from langgraph import graph as lg_graph
from langgraph import types as lg_types

from app import llm, rag, schemas
from app.core import config
from app.mcp_tools import contracts, provenance, runner, write_gate
from app.orchestration import prompting
from app.providers import outputs

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


class Outcome(pydantic.BaseModel):
    """Final, validated result of one graph run."""

    model_config = pydantic.ConfigDict(frozen=True)

    route: str
    answer: str
    citations: list[schemas.Citation] = []
    tool_calls: list[schemas.ToolCallTrace] = []
    abstained: bool = False
    grounding: str = "not_applicable"
    requires_confirmation: bool = False
    similarity: schemas.Similarity | None = None


def _merge(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {**left, **right}


class State(pydantic.BaseModel):
    """Graph state; list and timing fields are merged across parallel branches."""

    message: str
    confirm_action: bool = False
    classification: outputs.Classification | None = None
    chunks: Annotated[list[rag.RetrievedChunk], operator.add] = []
    tool_calls: Annotated[list[schemas.ToolCallTrace], operator.add] = []
    tool_results: Annotated[list[contracts.ToolOutput], operator.add] = []
    clarification: str = ""
    draft: outputs.Answer | None = None
    outcome: Outcome | None = None
    timings_ms: Annotated[dict[str, int], _merge] = {}


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
) -> outputs.Classification:
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
        role="classifier", prompt=prompt, schema=outputs.Classification
    )


def _format_chunks(chunks: list[rag.RetrievedChunk]) -> str:
    if not chunks:
        return "(ninguno)"
    return "\n\n".join(
        f"[chunk_id: {c.chunk_id} | doc_id: {c.doc_id} | title: {c.title}]\n{c.text}"
        for c in chunks
    )


def _format_tool_results(results: list[contracts.ToolOutput]) -> str:
    if not results:
        return "(ninguno)"
    return "\n".join(output.model_dump_json() for output in results)


def _template(state: State) -> Outcome:
    decision = state.classification
    assert decision is not None
    if decision.route == "injection":
        return Outcome(route="injection", answer="")
    if decision.route == "help":
        return Outcome(route="help", answer=HELP_ANSWER)
    if decision.route == "unsupported":
        return Outcome(
            route="unsupported", answer=UNSUPPORTED_ANSWER, abstained=True, grounding="ungrounded"
        )
    if not state.confirm_action:
        return Outcome(route="write", answer=WRITE_CONFIRMATION_ANSWER, requires_confirmation=True)
    return Outcome(route="write", answer=WRITE_UNAVAILABLE_ANSWER, abstained=True)


def _evidence_route(decision: outputs.Classification) -> str:
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
    draft = state.draft
    assert draft is not None and state.classification is not None
    chunks = state.chunks
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
        route=_evidence_route(state.classification),
        answer=draft.answer,
        citations=citations,
        tool_calls=state.tool_calls,
        abstained=abstained,
        grounding=draft.grounding,
        similarity=similarity,
    )


async def gather_tool_data(
    provider: llm.LLMProvider,
    tool_runner: runner.ToolRunner,
    settings: config.Settings,
    message: str,
    reference_date: datetime.date,
) -> dict[str, Any]:
    """Native tool-calling loop over the read tools, bounded by rounds and total calls.

    Every proposed call is checked against the allowlist, its schema and argument provenance
    before it runs. Rejected calls go back to the model as errors and never appear in traces.

    Args:
        provider: LLM backend for the tools role.
        tool_runner: Executes validated calls over MCP.
        settings: Limits for rounds and calls.
        message: User message.
        reference_date: Date that "hoy" refers to.

    Returns:
        State update with tool_calls, tool_results and, when nothing ran, a clarification.
    """
    prompt = prompting.load("tools").render(
        reference_date=reference_date.isoformat(), message=message
    )
    allowed = provenance.Provenance(
        text=message, dates=provenance.resolve_dates(message, reference_date)
    )
    specs = contracts.read_tool_specs()
    rounds: list[outputs.ToolRound] = []
    traces: list[schemas.ToolCallTrace] = []
    results: list[contracts.ToolOutput] = []
    clarification = ""
    calls_left = settings.max_tool_calls
    for _ in range(settings.max_tool_rounds):
        step = await provider.choose_tools(prompt=prompt, rounds=rounds, tools=specs)
        if not step.requests:
            clarification = step.text if not traces else ""
            break
        current: outputs.ToolRound = []
        for request in step.requests:
            try:
                if calls_left == 0:
                    raise ValueError("tool call limit reached")
                args = runner.validate_read(request.name, request.arguments)
                if missing := allowed.missing(args):
                    raise ValueError(f"not provided by the user: {', '.join(missing)}; ask for it")
            except ValueError as exc:
                current.append((request, {"error": str(exc)}))
                continue
            calls_left -= 1
            trace, result = await tool_runner.execute(request.name, args)
            allowed.add_result(result)
            traces.append(trace)
            results.append(contracts.ToolOutput(tool=request.name, result=result))
            current.append((request, result))
        rounds.append(current)
    return {"tool_calls": traces, "tool_results": results, "clarification": clarification}


def build(
    provider: llm.LLMProvider,
    settings: config.Settings,
    retriever: rag.Retriever | None = None,
    tool_runner: runner.ToolRunner | None = None,
) -> Any:
    """Compiles the answer graph.

    Args:
        provider: LLM backend for the classifier, tools and answer roles.
        settings: Application settings.
        retriever: Policy retriever; without one the policy branch is skipped.
        tool_runner: MCP tool runner; without one the tool and write branches are skipped.

    Returns:
        A compiled LangGraph runnable that takes and returns State.
    """
    reference = config.load_variant(settings.variant_path).reference_date
    reference_date = reference.isoformat()

    async def classify_node(state: State) -> dict[str, Any]:
        decision = await classify(provider, state.message, state.confirm_action)
        return {"classification": decision}

    def after_classify(state: State) -> list[lg_types.Send] | str:
        decision = state.classification
        assert decision is not None
        if decision.route != "answer" or (decision.write_requested and tool_runner is None):
            return "respond"
        if decision.write_requested:
            return "write"
        sends = []
        if decision.needs_tools and tool_runner is not None:
            sends.append(lg_types.Send("tools", state))
        if retriever is not None:
            sends.append(lg_types.Send("retrieve", state))
        return sends or "answer"

    async def write_node(state: State) -> dict[str, Any]:
        assert tool_runner is not None
        result = await write_gate.handle(state.message, state.confirm_action, tool_runner)
        outcome = Outcome(
            route="write",
            answer=result.answer,
            tool_calls=result.tool_calls,
            requires_confirmation=result.requires_confirmation,
        )
        return {"outcome": outcome}

    async def tools_node(state: State) -> dict[str, Any]:
        assert tool_runner is not None
        return await gather_tool_data(provider, tool_runner, settings, state.message, reference)

    async def respond_node(state: State) -> dict[str, Any]:
        return {"outcome": _template(state)}

    async def retrieve_node(state: State) -> dict[str, Any]:
        assert retriever is not None
        return {"chunks": await retriever.search(state.message, top_k=settings.rag_top_k)}

    async def answer_node(state: State) -> dict[str, Any]:
        chunks = state.chunks
        results = state.tool_results
        if not chunks and not results:
            draft = outputs.Answer(
                answer=state.clarification or NO_EVIDENCE_ANSWER,
                cited_chunk_ids=[],
                grounding="qualified" if state.clarification else "ungrounded",
            )
            return {"draft": draft}
        prompt = prompting.load("answer").render(
            reference_date=reference_date,
            chunks=_format_chunks(chunks),
            tool_results=_format_tool_results(results),
            missing=state.clarification or "(ninguno)",
            message=state.message,
        )
        draft = await provider.structured(role="answer", prompt=prompt, schema=outputs.Answer)
        return {"draft": draft}

    async def validate_node(state: State) -> dict[str, Any]:
        return {"outcome": validate(state)}

    builder = lg_graph.StateGraph(State)
    builder.add_node("classify", _timed("classify", classify_node))
    builder.add_node("respond", _timed("respond", respond_node))
    builder.add_node("write", _timed("write", write_node))
    builder.add_node("tools", _timed("tools", tools_node))
    builder.add_node("retrieve", _timed("retrieve", retrieve_node))
    builder.add_node("answer", _timed("answer", answer_node))
    builder.add_node("validate", _timed("validate", validate_node))
    builder.add_edge(lg_graph.START, "classify")
    builder.add_conditional_edges(
        "classify", after_classify, ["respond", "write", "tools", "retrieve", "answer"]
    )
    builder.add_edge("tools", "answer")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("write", lg_graph.END)
    builder.add_edge("answer", "validate")
    builder.add_edge("validate", lg_graph.END)
    builder.add_edge("respond", lg_graph.END)
    return builder.compile()
