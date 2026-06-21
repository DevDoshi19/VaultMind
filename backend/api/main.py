import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference

from app.graph import build_graph
from api.middleware.rate_limit import add_rate_limit_middleware
from api.routes import health, query

# ── Logging ───────────────────────────────────────────────────────────────────
# One logger for the whole API layer.
# Format: timestamp | level | logger name | message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vaultmind.api")

@asynccontextmanager
async def lifespan(app:FastAPI):
    # ── STARTUP ──
    logger.info("VaultMind backend starting up...")
    logger.info("Compiling LangGraph pipeline...")
 
    app.state.graph = build_graph()         # compiled once, reused forever
 
    logger.info("LangGraph pipeline ready.")
    logger.info("Backend is live — waiting for requests.")
 
    yield   # ← server runs here, handling requests
 
    # ── SHUTDOWN ──
    logger.info("VaultMind backend shutting down. Goodbye.")

app = FastAPI(
    title="VaultMind API",
    description="Production RAG backend for Dev Doshi's resume intelligence system.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# cors middleware so Streamlit / React can call us
ALLOWED_ORIGINS = [
    "http://localhost:8501",    # Streamlit dev
    "http://localhost:3000",    # React dev (future)
    "http://127.0.0.1:8501",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# rate limit middleware

add_rate_limit_middleware(app)

# Routers

app.include_router(health.router, prefix="",tags=["Health"])
app.include_router(query.router,  prefix="/api",tags=["Query"])
 

"""
The core idea of this file : 

The root of the VaultMind backend service.
Responsibilities:
  - Create the FastAPI app
  - Compile the LangGraph graph once at startup (lifespan)
  - Add CORS middleware so Streamlit / React can call us
  - Mount all routers (health, query)
"""

@app.get("/scalar", include_in_schema=False)
async def get_scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="Scalar API",
    )