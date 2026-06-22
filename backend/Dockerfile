# backend/Dockerfile
#
# Builds the VaultMind backend service.
# Contains: FastAPI + LangGraph pipeline + ChromaDB + BM25
#
# Build:  docker build -t vaultmind-backend .          (from backend/)
# Run:    docker run -p 8000:8000 vaultmind-backend    (standalone)
# Preferred: docker compose up                          (with frontend) (create both the image at once)

# -- Base image -------------------------------------------------------------------------
# python:3.11-slim =  ~130MB vs ~900MB for full python:3.11
# Slim strips docs, test files, and unused packages.

FROM python:3.11-slim

# -- Python env vars -------------------------------------------------------------------
# PYTHONDONTWRITEBYTECODE - skip .pyc files, saves space inside the container
# PYTHONUNBUFFERED        - flush stdout/stderr immediately, logs show in real time
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -- Working directory ----------------------------------------------------------------
# All subsequent commands run from /app inside the container.
# This is where your backend/ code will live.
WORKDIR /app

# -- System dependencies --------------------------------------------------------------
# build-essential — C compiler needed by some Python packages (chromadb, tokenizers)
# curl            — useful for health checks inside the container
# && rm -rf /var/lib/apt/lists/* — cleans apt cache, keeps image smaller
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -- Python dependencies --------------------------------------------------------------
# Copy requirements FIRST, install, THEN copy code.
# Reason: Docker caches each layer. If requirements.txt hasn't changed,
# this entire pip install step is skipped on rebuild — saves minutes.
# If you copy all code first, any code change invalidates this cache.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -- Application code -----------------------------------------------------------------
# Copy everything in backend/ into /app inside the container.
# .dockerignore (next file) prevents __pycache__, .env, chroma_db from being copied.
COPY . .

# -- Non-root user ----------------------------------------------------------------------
# Running as root inside a container is a security risk.
# If the app is exploited, attacker gets root — we limit that here.
RUN useradd --create-home --shell /bin/bash vaultmind
USER vaultmind

# -- Port --------------------------------------------------------------------------------
# Documents which port the container listens on.
# EXPOSE doesn't publish the port — docker-compose.yml does that.
EXPOSE 8000

# -- Healthcheck -------------------------------------------------------------------------
# Docker checks this every 30s. If it fails 3 times, container is marked unhealthy.
# Uses the shallow /health endpoint — just proves HTTP is responding.
# --interval  : how often to check
# --timeout   : how long to wait for a response
# --retries   : how many failures before marking unhealthy
# --start-period : grace period after startup before checks begin (graph compile takes ~2s)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

# -- Start command ----------------------------------------------------------------------
# Run uvicorn from inside /app (which is backend/).
# So "api.main:app" resolves to /app/api/main.py → app instance.
# --host 0.0.0.0  : listen on all interfaces inside the container
#                   (127.0.0.1 would only be accessible inside the container itself)
# --port 8000     : match EXPOSE above
# No --reload     : never use reload in production — restarts on any file change
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]