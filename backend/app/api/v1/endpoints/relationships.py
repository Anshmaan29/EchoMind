from fastapi import APIRouter, Depends, Query
from app.api.dependencies import get_knowledge_service
from app.knowledge.service import KnowledgeService
from app.schemas.relationship import RelationshipResponse

router = APIRouter(tags=["Relationships"])

@router.get("/relationships", response_model=list[RelationshipResponse], summary="List extracted relationships")
async def list_relationships(
    limit: int = Query(default=50, ge=1, le=200, description="Max items to return"),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
) -> list[RelationshipResponse]:
    """Retrieves list of extracted relationship edges connecting knowledge graph entities."""
    return await knowledge_service.list_relationships(limit=limit)
