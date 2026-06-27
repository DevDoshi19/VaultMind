"""
Rate limiting for the VaultMind API.
Protects the backend from abuse and runaway costs, every query hits OpenAI,
so an unbounded API is a credit card risk, not just a performance risk.
"""

from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse


# -- Limiter instance --
# get_remote_address is a SlowAPI helper that extracts the client IP from the request.
# This becomes the "key" of each unique IP gets its own counter.
# _default_limits applies to every route that uses this limiter unless overridden.

limiter = Limiter(
    key_func = get_remote_address,
    default_limits=["3/minute"]
)

# -- 429 handler --
# When the limit is exceeded, SlowAPI raises RateLimitExceeded.
# Without this handler, FastAPI would return a generic 500 error — confusing.
# This gives a clean, readable JSON response with the right status code.
async def _rate_limit_exceeded_handler(request:Request, exc:RateLimitExceeded)->JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error":"rate_limit_exceeded",
            "detail": f"Too many requests. LImit:{exc.limit} Try again in moment.",
            "retry_after_seconds": exc.retry_after, # tells client how long to wait
        }
    )

def add_rate_limit_middleware(app: FastAPI) -> None:
    """
    Attach SlowAPI to the FastAPI app.
    Called once in main.py — applies to every route automatically.
 
    Why a function instead of doing this directly in main.py?
    Keeps main.py clean. All rate limit config lives here, one place to change.
    """
    # attach the limiter to app state, route handlers access it via request.app.state.limiter
    app.state.limiter = limiter

    # register the 429 handler, SlowAPI raises RateLimitExceeded, FastAPI catches it here
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
 
    # added the middleware itself, this is what intercepts every request
    app.add_middleware(SlowAPIMiddleware)