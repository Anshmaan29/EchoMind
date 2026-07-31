import time
import uuid
from collections.abc import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware attaching X-Correlation-ID headers and logging HTTP request latency."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        start_time = time.perf_counter()

        response: Response = await call_next(request)

        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time-MS"] = str(process_time_ms)

        logger.info(
            f"HTTP {request.method} {request.url.path} -> {response.status_code} ({process_time_ms}ms)",
            correlation_id=correlation_id,
            status_code=response.status_code,
            duration_ms=process_time_ms
        )

        return response
