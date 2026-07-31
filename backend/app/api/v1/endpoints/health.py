from fastapi import APIRouter
from app.schemas.common import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    System Health Check Endpoint.
    Returns: {"status": "ok"}
    """
    return HealthResponse(status="ok")
