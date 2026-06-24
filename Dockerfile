FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .
RUN uv sync --no-dev --frozen

FROM python:3.13-slim

COPY --from=builder /app/.venv /app/.venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY src/ src/
COPY configs/ configs/
COPY outputs/ outputs/
COPY data/ data/

ENV PATH="/app/.venv/bin:$PATH"
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
