"""Forces offline backends so tests never make paid calls or download models, even with a .env."""

import os

os.environ["LLM_BACKEND"] = "fake"
os.environ["EMBEDDING_BACKEND"] = "fake"
os.environ["QDRANT_URL"] = ""
