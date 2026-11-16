"""Forces offline backends so tests never make paid calls, download models or send traces."""

import os

os.environ["LLM_BACKEND"] = "fake"
os.environ["EMBEDDING_BACKEND"] = "fake"
os.environ["QDRANT_URL"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
