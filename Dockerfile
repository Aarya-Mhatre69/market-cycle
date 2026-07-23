###############################################################################
# Stage 1 — Python dependency wheels
# Build wheels in a full image so the slim runtime never needs build tools.
###############################################################################
FROM python:3.12-slim AS python-builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first for layer caching.
COPY pyproject.toml .

# Export a plain requirements.txt from pyproject.toml and install into /install.
RUN pip install --no-cache-dir uv \
    && uv export --no-dev --no-hashes --format requirements-txt -o requirements.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


###############################################################################
# Stage 2 — Runtime image
###############################################################################
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Copy installed site-packages from builder.
COPY --from=python-builder /install /usr/local

# ── Application source ────────────────────────────────────────────────────────
COPY main.py          .
COPY src/             src/
COPY config/          config/

# ── Embed model artifacts directly into the image ────────────────────────────
# Models are small (~1.3 MB total) so we bake them in — no volume mount needed.
COPY models/          models/

EXPOSE 8000

# Graceful shutdown: uvicorn handles SIGTERM correctly.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
