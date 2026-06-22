# frontend/Dockerfile
#
# Builds the VaultMind frontend service.
# Contains: Streamlit + httpx only — no LangGraph, no ML libraries.
# This image is intentionally small — frontend is a thin HTTP client.
#
# Build:  docker build -t vaultmind-frontend .       (from frontend/)
# Preferred: docker compose up                        (with backend)

# ── Base image ────────────────────────────────────────────────────────────────
# Same slim base as backend for consistency.
# Frontend needs no C compiler or system libs — pip install is pure Python.
FROM python:3.11-slim

# ── Python env vars ───────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Only two packages — streamlit + httpx.
# No system deps needed unlike the backend (no chromadb, no tokenizers).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash vaultmind
USER vaultmind

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Streamlit exposes a /_stcore/health endpoint built-in — no custom route needed.
# start-period is shorter than backend — no graph compilation, starts in ~2s.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Streamlit config ──────────────────────────────────────────────────────────
# These env vars configure Streamlit without needing a config.toml file.
# STREAMLIT_SERVER_PORT           : match EXPOSE above
# STREAMLIT_SERVER_ADDRESS        : listen on all interfaces
# STREAMLIT_SERVER_HEADLESS       : true = no browser auto-open, correct for containers
# STREAMLIT_BROWSER_GATHER_USAGE_STATS : false = no telemetry pings on startup
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Start command ─────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "streamlit_app.py"]