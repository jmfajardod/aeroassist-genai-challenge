"""Per-request LLM token and cost accounting, collected through a context variable."""

from __future__ import annotations

import contextvars
from typing import Any

import pydantic

from app import schemas
from app.core import constants


class LlmCall(pydantic.BaseModel):
    """Token usage and estimated cost of one LLM call."""

    model_config = pydantic.ConfigDict(frozen=True)

    role: str
    model: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_usd: float | None


def estimate_cost(
    model: str, input_tokens: int, cached_tokens: int, output_tokens: int
) -> float | None:
    """Estimated cost in USD from the configured price table.

    Cached input tokens are a subset of input tokens and are billed at the cached rate; reasoning
    tokens are part of output tokens.

    Args:
        model: Model ID.
        input_tokens: All prompt tokens.
        cached_tokens: Prompt tokens served from the provider cache.
        output_tokens: Completion tokens, including reasoning.

    Returns:
        The cost, or None for a model without a price.
    """
    prices = constants.MODEL_PRICES_USD_PER_MILLION.get(model)
    if prices is None:
        return None
    billed = (
        (input_tokens - cached_tokens) * prices["input"]
        + cached_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    )
    return billed / 1_000_000


class UsageTracker:
    """Collects the LLM calls of one request."""

    def __init__(self) -> None:
        self.calls: list[LlmCall] = []

    def record(self, role: str, model: str, usage_metadata: dict[str, Any] | None) -> None:
        """Adds one call from a LangChain usage_metadata dict; ignores missing usage.

        Args:
            role: Model role (classifier, tools or answer).
            model: Model ID.
            usage_metadata: LangChain usage dict with input/output token counts.
        """
        if not usage_metadata:
            return
        input_tokens = int(usage_metadata.get("input_tokens", 0))
        output_tokens = int(usage_metadata.get("output_tokens", 0))
        details = usage_metadata.get("input_token_details") or {}
        cached = int(details.get("cache_read", 0) or 0)
        self.calls.append(
            LlmCall(
                role=role,
                model=model,
                input_tokens=input_tokens,
                cached_tokens=cached,
                output_tokens=output_tokens,
                cost_usd=estimate_cost(model, input_tokens, cached, output_tokens),
            )
        )

    def summary(self) -> schemas.Usage | None:
        """Totals for the request, or None when no LLM call reported usage."""
        if not self.calls:
            return None
        costs = [call.cost_usd for call in self.calls]
        return schemas.Usage(
            llm_calls=len(self.calls),
            input_tokens=sum(call.input_tokens for call in self.calls),
            cached_tokens=sum(call.cached_tokens for call in self.calls),
            output_tokens=sum(call.output_tokens for call in self.calls),
            cost_usd=round(sum(c for c in costs if c is not None), 6)
            if all(c is not None for c in costs)
            else None,
        )


_current: contextvars.ContextVar[UsageTracker | None] = contextvars.ContextVar(
    "usage_tracker", default=None
)


def start() -> UsageTracker:
    """Starts tracking for the current request and returns the tracker."""
    tracker = UsageTracker()
    _current.set(tracker)
    return tracker


def record(role: str, model: str, usage_metadata: dict[str, Any] | None) -> None:
    """Records a call on the current request's tracker, if one is active."""
    tracker = _current.get()
    if tracker is not None:
        tracker.record(role, model, usage_metadata)
