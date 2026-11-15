"""MCP transport behind an interface; the stdio implementation owns one long-lived session.

The session lives on a dedicated event-loop thread, so it survives across request event loops
and calls are serialized through a single lock. Any failure or timeout restarts the session.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import threading
from typing import Any, Protocol

import mcp
from mcp.client import stdio

from app.core import errors

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


class McpTransport(Protocol):
    """Calls tools on an MCP server; a remote HTTP transport can replace stdio in production."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Calls one tool and returns its decoded result.

        Args:
            name: Tool name.
            arguments: Already validated arguments.

        Returns:
            The decoded tool result.

        Raises:
            errors.UpstreamError: If the server fails or rejects the call.
            errors.UpstreamTimeoutError: If the call times out.
        """

    async def list_tools(self) -> list[str]:
        """Returns the names of the tools the server exposes."""


class StdioMcpTransport:
    """Spawns the provided MCP server as a subprocess and talks to it over stdio.

    The subprocess gets the SDK's minimal default environment, so app secrets are not passed.

    Args:
        call_timeout_seconds: Limit for each tool call.
        startup_timeout_seconds: Limit for spawning and initializing the server.
    """

    def __init__(self, call_timeout_seconds: float, startup_timeout_seconds: float) -> None:
        self._params = mcp.StdioServerParameters(
            command=sys.executable, args=["-m", "mcp_server.server"], cwd=_PROJECT_ROOT
        )
        self._call_timeout = call_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._thread_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self._owner: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._session: mcp.ClientSession | None = None

    def _worker_loop(self) -> asyncio.AbstractEventLoop:
        with self._thread_lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._loop.run_forever, name="mcp-stdio", daemon=True
                ).start()
            return self._loop

    async def _own_session(
        self, ready: asyncio.Future[mcp.ClientSession], stop: asyncio.Event
    ) -> None:
        """Holds the stdio and session contexts open in one task until asked to stop."""
        try:
            with open(os.devnull, "w") as devnull:
                async with (
                    stdio.stdio_client(self._params, errlog=devnull) as (read, write),
                    mcp.ClientSession(read, write) as session,
                ):
                    await session.initialize()
                    ready.set_result(session)
                    await stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)

    async def _ensure_session(self) -> mcp.ClientSession:
        if self._session is not None and self._owner is not None and not self._owner.done():
            return self._session
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[mcp.ClientSession] = loop.create_future()
        self._stop = asyncio.Event()
        self._owner = loop.create_task(self._own_session(ready, self._stop))
        self._session = await asyncio.wait_for(ready, self._startup_timeout)
        return self._session

    async def _restart(self) -> None:
        self._session = None
        if self._stop is not None and self._owner is not None:
            self._stop.set()
            await asyncio.wait({self._owner}, timeout=2)

    async def _run(self, name: str | None, arguments: dict[str, Any]) -> Any:
        """Runs one serialized request on the worker loop; name None lists the tools."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                session = await self._ensure_session()
                if name is None:
                    return await asyncio.wait_for(session.list_tools(), self._call_timeout)
                return await asyncio.wait_for(
                    session.call_tool(name, arguments), self._call_timeout
                )
            except BaseException:
                await self._restart()
                raise

    async def _submit(self, name: str | None, arguments: dict[str, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(self._run(name, arguments), self._worker_loop())
        try:
            return await asyncio.wrap_future(future)
        except TimeoutError as exc:
            raise errors.UpstreamTimeoutError("The tool server timed out.") from exc
        except Exception as exc:
            raise errors.UpstreamError("The tool server failed.") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Calls one tool and decodes its JSON result.

        Args:
            name: Tool name.
            arguments: Already validated arguments.

        Returns:
            The decoded tool result.

        Raises:
            errors.UpstreamTimeoutError: If the call timed out.
            errors.UpstreamError: If the server failed or returned a tool error.
        """
        result = await self._submit(name, arguments)
        if result.isError or not result.content:
            raise errors.UpstreamError("The tool server rejected the call.")
        return json.loads(result.content[0].text)

    async def list_tools(self) -> list[str]:
        """Returns the tool names the server exposes; used by readiness."""
        result = await self._submit(None, {})
        return [tool.name for tool in result.tools]
