"""Forces the offline LLM backend so tests never make paid calls, even with a local .env."""

import os

os.environ["LLM_BACKEND"] = "fake"
