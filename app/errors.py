"""Typed application errors and the JSON error envelope shared by every endpoint."""

from __future__ import annotations

import fastapi
from fastapi import exceptions, responses
from starlette import exceptions as starlette_exceptions


class AppError(Exception):
    """Base error whose message is safe to return to clients."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsafeRequestError(AppError):
    """The request is well-formed but unsafe or semantically invalid."""

    status_code = 400
    code = "unsafe_request"


class PromptInjectionError(AppError):
    """The message tries to override instructions or exfiltrate data."""

    status_code = 400
    code = "prompt_injection"


class RateLimitedError(AppError):
    """The local concurrency or rate limit is exhausted."""

    status_code = 429
    code = "rate_limited"


class UpstreamError(AppError):
    """An LLM, tool or retrieval dependency failed after retries."""

    status_code = 502
    code = "upstream_error"


class NotReadyError(AppError):
    """A required dependency is not configured or not initialized."""

    status_code = 503
    code = "not_ready"


class UpstreamTimeoutError(AppError):
    """The request exceeded its overall time budget."""

    status_code = 504
    code = "upstream_timeout"


_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


def error_response(
    status_code: int, code: str, message: str, request_id: str | None
) -> responses.JSONResponse:
    """Builds the contract error envelope.

    Args:
        status_code: HTTP status to return.
        code: Stable machine-readable error code.
        message: Client-safe description.
        request_id: Request ID of the failing request, if known.

    Returns:
        A JSON response {"error": {"code", "message"}, "request_id"}.
    """
    return responses.JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}, "request_id": request_id},
    )


def _request_id(request: fastapi.Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def _app_error(request: fastapi.Request, exc: Exception) -> responses.JSONResponse:
    assert isinstance(exc, AppError)
    return error_response(exc.status_code, exc.code, exc.message, _request_id(request))


async def _validation_error(request: fastapi.Request, exc: Exception) -> responses.JSONResponse:
    """Summarizes schema errors by field location without echoing the submitted input."""
    assert isinstance(exc, exceptions.RequestValidationError)
    details = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'] if part != 'body')}: {error['msg']}"
        for error in exc.errors()
    )
    return error_response(422, "validation_error", details, _request_id(request))


async def _http_error(request: fastapi.Request, exc: Exception) -> responses.JSONResponse:
    assert isinstance(exc, starlette_exceptions.HTTPException)
    code = _HTTP_CODES.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, str(exc.detail), _request_id(request))


def install_handlers(application: fastapi.FastAPI) -> None:
    """Registers envelope-producing handlers for app, schema and routing errors.

    Args:
        application: The FastAPI app to configure.
    """
    application.add_exception_handler(AppError, _app_error)
    application.add_exception_handler(exceptions.RequestValidationError, _validation_error)
    application.add_exception_handler(starlette_exceptions.HTTPException, _http_error)
