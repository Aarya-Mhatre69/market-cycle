###############################################################################
# Stage 1 — build: install dependencies into a separate prefix
# Using a full image here so gcc/build-essential are available for native exts.
###############################################################################
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv, export deps from pyproject.toml, then pip-install into /install
# so the runtime stage gets a clean, minimal copy.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv export --locked --no-dev --no-hashes --format requirements-txt -o requirements.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


###############################################################################
# Stage 2 — runtime: lean image, no build tools
###############################################################################
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Pull in the installed packages from the builder stage.
COPY --from=builder /install /usr/local

# Application source
COPY main.py   .
COPY src/      src/
COPY config/   config/

# Embed model artifacts — all files total ~1.3 MB, safe to bake in.
COPY models/   models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
