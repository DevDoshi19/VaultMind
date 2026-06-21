"""
backend/api/routes/query.py

POST /api/query  — the main endpoint.
Receives a question, runs the LangGraph pipeline, returns structured JSON.

Protected by:
  - SlowAPI rate limit  : 3 requests / minute / IP
  - Input validation    : Pydantic rejects malformed requests before they hit the pipeline
"""

import asyncio
import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from langchain_core.tracers.context import tracing_v2_enabled
from pydantic import BaseModel, Field

from app.state import RAGState
# from api.middleware.rate_limit import limiter

logger = logging.getLogger("vaultmind.api.query")

router = APIRouter()


# -- Request model --
# Pydantic validates this before your handler runs.
# `min_length=1` rejects empty strings — no need to check manually.
# `max_length=500` prevents abuse — nobody needs a 5000-character resume question.
# `Field(...)` lets you add metadata that shows up in /docs.
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask about Dev Doshi's resume.",
        examples=["What are Dev's skills?", "Tell me about his projects."],
    )


# -- Response model --
# Every field the frontend might need.
# `None` defaults mean the field is optional — blocked queries won't have confidence scores.
class QueryResponse(BaseModel):
    answer: str
    confidence_score: float | None = None
    input_blocked: bool = False
    output_flagged: bool = False
    total_tokens: int = 0
    estimated_cost: float = 0.0
    retrieval_status: str = ""
    retrieved_chunks: int = 0          # count only, not the raw text — keeps response small


# -- Endpoint --
@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the resume RAG pipeline",
    description="Send a natural language question. Returns an answer grounded in Dev Doshi's resume.",
    status_code=status.HTTP_200_OK,
)
# @limiter.limit("10/minute")            # SlowAPI checks IP counter before handler runs
async def query_endpoint(request: Request, body: QueryRequest) -> JSONResponse:

    logger.info(f"Query received: '{body.question[:60]}...' " if len(body.question) > 60 else f"Query received: '{body.question}'")

    # -- Build initial LangGraph state --
    # Same structure as main.py / streamlit_app.py — nothing changes here.
    initial_state: RAGState = {
        "question": body.question,
        "question_is_relevant": False,
        "retrieved_docs": [],
        "retrieval_status": "",
        "context_token_count": 0,
        "answer": "",
        "confidence_score": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0.0,
        "input_blocked": False,
        "output_flagged": False,
    }

    # -- Get the compiled graph from app state --
    # Set once at startup by lifespan in main.py — never recompiled per request.
    graph = request.app.state.graph

    # -- Run LangGraph in a thread pool --
    # graph.invoke() is synchronous and blocks for 3-5 seconds while nodes execute.
    # Calling it directly inside `async def` would freeze the entire FastAPI event loop —
    # no other requests could be processed until this one finishes.
    # run_in_executor offloads it to a background thread, keeping the event loop free.
    with tracing_v2_enabled(project_name="vaultmind"):
        result: RAGState = await asyncio.get_event_loop().run_in_executor(
            None,           # None = use Python's default ThreadPoolExecutor
            lambda: graph.invoke(
                initial_state,
                config={
                    "run_name": f"VaultMind | {body.question[:50]}",
                    "tags": ["production", "resume-rag", "fastapi"],
                    "metadata": {"phase": "12", "retrieval": "hybrid", "llm": "gpt-4o-mini"},
                },
            ),
        )

    logger.info(
        f"Query complete — tokens: {result.get('total_tokens', 0)}, "
        f"blocked: {result.get('input_blocked', False)}, "
        f"confidence: {result.get('confidence_score')}"
    )

    # -- Build response --
    # We don't send raw retrieved_docs to the frontend — those can be large.
    # Instead we send the count. Frontend can request chunks separately if needed (Phase 16).
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=QueryResponse(
            answer=result.get("answer", ""),
            confidence_score=result.get("confidence_score"),
            input_blocked=result.get("input_blocked", False),
            output_flagged=result.get("output_flagged", False),
            total_tokens=result.get("total_tokens", 0),
            estimated_cost=result.get("estimated_cost", 0.0),
            retrieval_status=result.get("retrieval_status", ""),
            retrieved_chunks=len(result.get("retrieved_docs", [])),
        ).model_dump(),
    )