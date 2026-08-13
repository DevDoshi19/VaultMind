"""
backend/api/routes/query.py

POST /api/query  — the main endpoint.
Receives a question, runs the LangGraph pipeline, returns structured JSON.

Redis caching added in Phase 15b:
  - Cache key: md5(question.lower().strip())
  - Cache hit: return cached response instantly, zero LLM calls
  - Cache miss: run full pipeline, store result with 24hr TTL
  - Redis failure: log warning and fall through to pipeline (never crash)

Protected by:
  - SlowAPI rate limit  : 3 requests / minute / IP
  - Input validation    : Pydantic rejects malformed requests before they hit the pipeline
"""

import asyncio
import hashlib
import json
import logging

import redis
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from langchain_core.tracers.context import tracing_v2_enabled
from pydantic import BaseModel, Field

from app.config import settings
from app.state import RAGState

logger = logging.getLogger("vaultmind.api.query")

router = APIRouter()


# -- Request model --
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask about Dev Doshi's resume.",
        examples=["What are Dev's skills?", "Tell me about his projects."],
    )


# -- Response model --
class QueryResponse(BaseModel):
    answer: str
    confidence_score: float | None = None
    input_blocked: bool = False
    output_flagged: bool = False
    total_tokens: int = 0
    estimated_cost: float = 0.0
    retrieval_status: str = ""
    retrieved_chunks: int = 0
    cached: bool = False           


# -- Cache helpers --
def _cache_key(question: str) -> str:
    # Normalize the question before hashing so "What are his skills?"
    # and "what are his skills?" hit the same cache entry.
    normalized = question.lower().strip()
    return f"vaultmind:query:{hashlib.md5(normalized.encode()).hexdigest()}"


def _get_redis_client():
    # We create a fresh client per request — Redis client is lightweight.
    # In Phase 16 we'll move this to a connection pool on app.state.
    return redis.from_url(settings.redis_url, decode_responses=True)


def _get_cached_response(question: str) -> dict | None:
    # Returns cached response dict if found, None on miss or Redis failure.
    # Redis failure is non-fatal — we always fall through to the pipeline.
    try:
        client = _get_redis_client()
        key = _cache_key(question)
        cached = client.get(key)
        if cached:
            logger.info(f"⚡ Cache hit for question: '{question[:50]}'")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis get failed — falling through to pipeline: {e}")
    return None


def _set_cached_response(question: str, response: dict) -> None:
    # Stores response in Redis with 24hr TTL.
    # Failure is non-fatal — cache miss on next request is acceptable.
    try:
        client = _get_redis_client()
        key = _cache_key(question)
        client.setex(
            name=key,
            time=86400,            # 24 hours in seconds
            value=json.dumps(response),
        )
        logger.info(f"Cached response for question: '{question[:50]}'")
    except Exception as e:
        logger.warning(f"Redis set failed — response not cached: {e}")


# -- Endpoint --
@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the resume RAG pipeline",
    description="Send a natural language question. Returns an answer grounded in Dev Doshi's resume.",
    status_code=status.HTTP_200_OK,
)
async def query_endpoint(request: Request, body: QueryRequest) -> JSONResponse:

    logger.info(
        f"Query received: '{body.question[:60]}...'"
        if len(body.question) > 60
        else f"Query received: '{body.question}'"
    )

    # -- Cache check --
    # Check Redis before touching the pipeline.
    # Cache hit = instant response, zero OpenAI calls, zero cost.
    cached = _get_cached_response(body.question)
    if cached:
        cached["cached"] = True
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=cached,
        )

    # -- Build initial LangGraph state --
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

    graph = request.app.state.graph

    # -- Run LangGraph pipeline --
    with tracing_v2_enabled(project_name="vaultmind"):
        result: RAGState = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: graph.invoke(
                initial_state,
                config={
                    "run_name": f"VaultMind | {body.question[:50]}",
                    "tags": ["production", "resume-rag", "fastapi"],
                    "metadata": {"phase": "15b", "retrieval": "hybrid", "llm": "gpt-4o-mini"},
                },
            ),
        )

    logger.info(
        f"Query complete — tokens: {result.get('total_tokens', 0)}, "
        f"blocked: {result.get('input_blocked', False)}, "
        f"confidence: {result.get('confidence_score')}"
    )

    # -- Build response --
    response = QueryResponse(
        answer=result.get("answer", ""),
        confidence_score=result.get("confidence_score"),
        input_blocked=result.get("input_blocked", False),
        output_flagged=result.get("output_flagged", False),
        total_tokens=result.get("total_tokens", 0),
        estimated_cost=result.get("estimated_cost", 0.0),
        retrieval_status=result.get("retrieval_status", ""),
        retrieved_chunks=len(result.get("retrieved_docs", [])),
        cached=False,
    ).model_dump()

    # -- Store in cache --
    # Only cache successful, non-blocked responses.
    # Blocked queries are cheap (no retrieval) so caching them isn't worth it.
    if not result.get("input_blocked", False):
        _set_cached_response(body.question, response)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response,
    )