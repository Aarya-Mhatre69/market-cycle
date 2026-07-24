# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install only third-party dependencies first
COPY pyproject.toml uv.lock ./

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project \
    --no-cache

# Copy application
COPY main.py .
COPY src/ src/
COPY config/ config/
COPY models/ models/

# Install the project itself
RUN uv sync \
    --frozen \
    --no-dev \
    --no-cache

# Create non-root user
RUN useradd --system --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]