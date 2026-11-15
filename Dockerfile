FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY app ./app
COPY mcp_server ./mcp_server
RUN uv sync --frozen --no-default-groups --no-editable

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PORT=8080

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY --from=builder /app/.venv ./.venv
COPY app ./app
COPY mcp_server ./mcp_server
COPY knowledge_base ./knowledge_base
COPY config ./config

USER app
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
