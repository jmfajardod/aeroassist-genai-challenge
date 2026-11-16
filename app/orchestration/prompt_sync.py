"""Pushes the bundled prompts to Langfuse Prompt Management as new versions when they changed.

Usage: python -m app.orchestration.prompt_sync
"""

from __future__ import annotations

import json
import sys

from app.core import config
from app.observability import tracing
from app.orchestration import prompting

PROMPT_NAMES = ("classifier", "tools", "answer")


def main() -> int:
    """Creates a labelled Langfuse version for each bundled prompt whose text differs.

    Returns:
        Process exit code; 1 when Langfuse is not configured.
    """
    settings = config.get_settings()
    client = tracing.Tracer(settings).client
    if client is None:
        print("Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY).")
        return 1
    report = {}
    for name in PROMPT_NAMES:
        bundled = prompting.load(name)
        messages = [
            {"role": "system", "content": bundled.system},
            {"role": "user", "content": bundled.user_template},
        ]
        try:
            current = client.get_prompt(
                name, type="chat", label=settings.langfuse_prompt_label, cache_ttl_seconds=0
            )
            unchanged = [(m["role"], m["content"]) for m in current.prompt] == [
                (m["role"], m["content"]) for m in messages
            ]
        except Exception:
            unchanged = False
        if unchanged:
            report[name] = "unchanged"
            continue
        created = client.create_prompt(
            name=name,
            type="chat",
            prompt=messages,
            labels=[settings.langfuse_prompt_label],
            commit_message=f"bundled {bundled.version}",
        )
        report[name] = f"created v{created.version}"
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
