from fastapi import APIRouter, Depends, Query
from app.api.dependencies import get_knowledge_service
from app.knowledge.service import KnowledgeService
from app.schemas.graph import GraphSearchResponse, KnowledgeGraphResponse

router = APIRouter(tags=["Knowledge Graph"])

@router.get("/graph/search", response_model=GraphSearchResponse, summary="Search knowledge graph")
async def search_graph(
    q: str = Query(..., min_length=1, description="Search query term or entity name"),
    limit: int = Query(default=10, ge=1, le=50, description="Max matched entities"),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
) -> GraphSearchResponse:
    """Performs hybrid graph search returning matching entities and subgraphs."""
    return await knowledge_service.search_graph(query=q, limit=limit)

@router.get(
    "/graph/neighbors/{entity_id}",
    response_model=KnowledgeGraphResponse,
    summary="Get entity N-hop graph neighborhood"
)
async def get_graph_neighbors(
    entity_id: str,
    depth: int = Query(default=1, ge=1, le=3, description="N-hop traversal depth"),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
) -> KnowledgeGraphResponse:
    """Retrieves N-hop graph neighborhood entities and edges surrounding an entity."""
    return await knowledge_service.get_neighbors(entity_id=entity_id, depth=depth)
