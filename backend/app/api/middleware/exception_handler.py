from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.exceptions import EchoMindException
from app.core.logging import logger

async def custom_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Centralized exception handler mapping domain errors to JSON responses."""
    if isinstance(exc, EchoMindException):
        logger.warning(
            f"Domain Exception on {request.method} {request.url.path}: {exc.message}",
            status_code=exc.status_code,
            details=exc.details
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.__class__.__name__,
                    "details": exc.details
                }
            }
        )

    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An internal server error occurred.",
                "type": "InternalServerError"
            }
        }
    )
