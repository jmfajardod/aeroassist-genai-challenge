"""Prompt templates stored next to the code.

Each prompt is a static system message followed by a user message that holds every variable
part, so the shared prefix stays cacheable by the provider.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import pathlib
import re
from collections.abc import Callable
from typing import Any

import pydantic

logger = logging.getLogger("aeroassist")

PromptLoader = Callable[[str], "Prompt"]

_PROMPTS_DIR = pathlib.Path(__file__).resolve().parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class RenderedPrompt(pydantic.BaseModel):
    """A prompt ready to send, plus the variables used to build it.

    Attributes:
        name: Prompt name.
        version: Short content hash of the templates.
        system: System message.
        user: Rendered user message.
        variables: Raw values substituted into the user template.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    version: str
    system: str
    user: str
    variables: dict[str, str]


class Prompt(pydantic.BaseModel):
    """A system message and a user template."""

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    system: str
    user_template: str

    @property
    def version(self) -> str:
        """Short content hash identifying this exact prompt text."""
        digest = hashlib.sha256(f"{self.system}\0{self.user_template}".encode())
        return digest.hexdigest()[:12]

    def render(self, **variables: str) -> RenderedPrompt:
        """Fills the user template.

        Args:
            **variables: Value for every placeholder in the user template.

        Returns:
            The rendered prompt.

        Raises:
            KeyError: If a placeholder has no value.
        """
        user = _PLACEHOLDER.sub(lambda match: variables[match.group(1)], self.user_template)
        return RenderedPrompt(
            name=self.name, version=self.version, system=self.system, user=user, variables=variables
        )


@functools.cache
def load(name: str) -> Prompt:
    """Loads a prompt by name from the bundled templates.

    Args:
        name: Prompt name, e.g. "classifier".

    Returns:
        The prompt.
    """
    system = (_PROMPTS_DIR / f"{name}_system.md").read_text(encoding="utf-8").strip()
    user = (_PROMPTS_DIR / f"{name}_user.md").read_text(encoding="utf-8").strip()
    return Prompt(name=name, system=system, user_template=user)


def load_from_langfuse(client: Any, label: str, name: str) -> Prompt:
    """Fetches a prompt from Langfuse Prompt Management, falling back to the bundled copy.

    The SDK caches prompts for 60 seconds and serves the fallback when the fetch fails, so a
    Langfuse outage never breaks a request.

    Args:
        client: Langfuse client.
        label: Prompt label to serve, e.g. "production".
        name: Prompt name.

    Returns:
        The remote prompt, or the bundled one if it cannot be fetched or is malformed.
    """
    bundled = load(name)
    fallback = [
        {"role": "system", "content": bundled.system},
        {"role": "user", "content": bundled.user_template},
    ]
    try:
        remote = client.get_prompt(
            name,
            type="chat",
            label=label,
            cache_ttl_seconds=60,
            fallback=fallback,
            max_retries=1,
            fetch_timeout_seconds=3,
        )
        by_role = {message["role"]: message["content"] for message in remote.prompt}
        return Prompt(name=name, system=by_role["system"], user_template=by_role["user"])
    except Exception as exc:
        logger.warning("prompt_fallback", extra={"prompt": name, "error": type(exc).__name__})
        return bundled
