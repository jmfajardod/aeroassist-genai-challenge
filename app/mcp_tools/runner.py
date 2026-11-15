"""Allowlisted, validated execution of MCP tool calls with an auditable trace per call."""

from __future__ import annotations

import time
from typing import Any, TypeVar

import pydantic

from app import schemas
from app.core import constants, errors
from app.mcp_tools import contracts, transport

ResultT = TypeVar("ResultT", bound=contracts.ToolResult)


def _status(result: dict[str, Any]) -> str:
    if result.get("found") is False:
        return "not_found"
    if result.get("created") is False:
        return "rejected"
    return "ok"


class ToolRunner:
    """Validates and executes tool calls over MCP and records an auditable trace for each.

    Args:
        mcp: MCP transport.
    """

    def __init__(self, mcp: transport.McpTransport) -> None:
        self._mcp = mcp

    async def execute(
        self, name: str, args: pydantic.BaseModel
    ) -> tuple[schemas.ToolCallTrace, dict[str, Any]]:
        """Calls one already validated tool.

        Args:
            name: Allowlisted tool name.
            args: Validated arguments.

        Returns:
            The trace and the decoded result.
        """
        if name not in constants.TOOL_NAMES:
            raise errors.UnsafeRequestError(f"Tool '{name}' is not allowed.")
        arguments = args.model_dump(mode="json")
        started = time.perf_counter()
        result = await self._mcp.call_tool(name, arguments)
        trace = schemas.ToolCallTrace(
            name=name,
            arguments=arguments,
            status=_status(result),
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        return trace, result

    async def execute_typed(
        self, name: str, args: pydantic.BaseModel, result_type: type[ResultT]
    ) -> tuple[schemas.ToolCallTrace, ResultT]:
        """Calls one tool and validates its untrusted result against result_type.

        Args:
            name: Allowlisted tool name.
            args: Validated arguments.
            result_type: Model the result must match.

        Returns:
            The trace and the validated result.

        Raises:
            errors.UpstreamError: If the result does not match the expected shape.
        """
        trace, result = await self.execute(name, args)
        try:
            return trace, result_type.model_validate(result)
        except pydantic.ValidationError as exc:
            raise errors.UpstreamError("The tool server returned an unexpected result.") from exc


def validate_read(name: str, raw: dict[str, Any]) -> pydantic.BaseModel:
    """Validates a model-proposed read call against the allowlist and schema.

    Args:
        name: Proposed tool name.
        raw: Proposed arguments.

    Returns:
        The validated arguments.

    Raises:
        ValueError: If the tool is not an allowed read tool or the arguments are invalid.
    """
    if name not in contracts.READ_TOOLS:
        raise ValueError(f"tool '{name}' is not available")
    try:
        return contracts.READ_TOOLS[name].model_validate(raw)
    except pydantic.ValidationError as exc:
        raise ValueError(str(exc.errors(include_url=False, include_input=False))) from exc
