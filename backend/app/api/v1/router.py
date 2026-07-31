from fastapi import APIRouter
from app.api.v1.endpoints import documents, entities, graph, health, relationships, timeline

api_v1_router = APIRouter()

# Include endpoint sub-routers
api_v1_router.include_router(health.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(entities.router)
api_v1_router.include_router(relationships.router)
api_v1_router.include_router(timeline.router)
api_v1_router.include_router(graph.router)
