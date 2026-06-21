"""
GET /health           → shallow probe  (is the HTTP server alive?)
GET /health?deep=true → deep probe     (is the LangGraph pipeline ready?)
 
Used by:
  - Docker HEALTHCHECK instruction       
  - Kubernetes liveness + readiness      
  - You, manually, to verify startup
"""

import time 
from datetime import datetime,timezone

from fastapi import APIRouter, Request,status
from fastapi.responses import JSONResponse
from pydantic import BaseModel


router = APIRouter()

# Track when the server started — used to compute uptime ,module-level variable, set once when Python imports this file.
_SERVER_START_TIME = time.time()

class HealthResponse(BaseModel):
    status: str                   # "ok" or "degraded" (degraded means HTTP is up but graph failed)
    version: str                  # API version string, useful when you have multiple deployments and upgreadge 
    uptime_sec: float             # seconds since server started
    timestamp: str                # ISO-8601 UTC string, useful for log correlation
    pipeline: dict | None = None  # None for shallow probe, dict for deep probe

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Shallow or deep health probe. Use ?deep=true for readiness checks."
)
def health_check(request : Request,deep:bool=False)->JSONResponse:
    uptime = round(time.time() - _SERVER_START_TIME, 2)
    timestamp = datetime.now(timezone.utc).isoformat()
 
    # -- Shallow probe (it is like scanning )-- 
    # Just proves the HTTP server is alive and responding.
    # Returns immediately,, no dependency checks and no deep check it's just like checking a patient from outside .
    # the main use is : Docker liveness probe uses this.
    if not deep:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=HealthResponse(
                status="ok",
                version="1.0.0",
                uptime_sec=uptime,
                timestamp=timestamp,
            ).model_dump(),
        )

    # -- Deep probe --
    # Checks that the LangGraph pipeline compiled successfully at startup or not 
    # `request.app.state` is the same object lifespan wrote to in main file so If `graph` attribute is missing, startup failed silently — return 503.
    # HTTP 503 = Service Unavailable.
    # Kubernetes readiness probe: if 503, don't route traffic here. This prevents users from hitting a container that started but is broken.
    
    # baiscally Deep probe means check everything in deep , ex. is graph has build , langgraph is working etc.
    
    pipeline_info = None
    http_status = status.HTTP_200_OK
    health_status = "ok"
 
    # lifespan may have failed before setting this defensive access
    graph = getattr(request.app.state, "graph", None)
 
    if graph is None:
        # Graph didn't compile or something went wrong in lifespan startup so we return the degraded status
        health_status = "degraded"
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE    # 503 tells Kubernetes: don't route traffic here yet
        pipeline_info = {"ready": False, "reason": "LangGraph pipeline not compiled"}
    else:
        # Graph exists, report basic info about it 
        # graph.nodes is a dict of node_name → node_function
        pipeline_info = {
            "ready": True,  
            "nodes": list(graph.nodes.keys()),   # e.g. ["input_guardrail", "classify", ...]
            "node_count": len(graph.nodes),
        }
 

    return JSONResponse(
        status_code=http_status,
        content=HealthResponse(
            status=health_status,
            version="1.0.0",
            uptime_sec=uptime,
            timestamp=timestamp,
            pipeline=pipeline_info,
        ).model_dump(),
    )

