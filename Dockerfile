# ── Stage 1: Builder — install all Python deps into a venv ──────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (maximises Docker layer cache)
COPY dashboard_app/backend/requirements.txt .

# Install into /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model so the container starts instantly
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built venv and model cache from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache /root/.cache

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source
COPY dashboard_app/backend/ ./backend/
COPY dashboard_app/frontend/ ./frontend/

# Expose FastAPI port
EXPOSE 8000

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/backend/bert_model && \
    chown -R appuser:appuser /root/.cache 2>/dev/null || true
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

# Start FastAPI (no --reload in production)
CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
