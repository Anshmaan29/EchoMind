from fastapi import APIRouter, Depends, Query
from app.api.dependencies import get_timeline_service
from app.schemas.timeline import TimelineResponse
from app.timeline.service import TimelineService

router = APIRouter(tags=["Timeline Engine"])

@router.get("/timeline", response_model=TimelineResponse, summary="Get global chronological timeline")
async def get_timeline(
    limit: int = Query(default=50, ge=1, le=200, description="Max timeline events to return"),
    timeline_service: TimelineService = Depends(get_timeline_service)
) -> TimelineResponse:
    """Retrieves chronological timeline events across all ingested memories."""
    return await timeline_service.get_timeline(limit=limit)

@router.get(
    "/timeline/project/{project_name}",
    response_model=TimelineResponse,
    summary="Get project evolution timeline replay"
)
async def get_project_timeline(
    project_name: str,
    limit: int = Query(default=50, ge=1, le=200, description="Max events"),
    timeline_service: TimelineService = Depends(get_timeline_service)
) -> TimelineResponse:
    """Retrieves project evolution timeline replay for a specific project."""
    return await timeline_service.get_project_timeline(project_name=project_name, limit=limit)
