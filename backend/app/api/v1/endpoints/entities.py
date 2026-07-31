from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.api.dependencies import get_knowledge_service
from app.knowledge.service import KnowledgeService
from app.schemas.entity import EntityResponse

router = APIRouter(tags=["Entities"])

@router.get("/entities", response_model=list[EntityResponse], summary="List extracted entities")
async def list_entities(
    type: str | None = Query(default=None, description="Optional entity type filter"),
    limit: int = Query(default=50, ge=1, le=200, description="Max items to return"),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
) -> list[EntityResponse]:
    """Retrieves list of extracted domain entities with optional type filtering."""
    return await knowledge_service.list_entities(entity_type=type, limit=limit)

@router.get("/entities/{id}", response_model=EntityResponse, summary="Get entity details by ID")
async def get_entity_by_id(
    id: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
) -> EntityResponse:
    """Retrieves a specific entity record by ID."""
    entity = await knowledge_service.get_entity_by_id(entity_id=id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity with ID '{id}' was not found."
        )
    return entity
