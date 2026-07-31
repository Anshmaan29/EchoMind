from typing import Any
from pydantic import BaseModel
from app.schemas.entity import EntityResponse
from app.schemas.relationship import RelationshipResponse

class KnowledgeGraphResponse(BaseModel):
    entities: list[EntityResponse]
    relationships: list[RelationshipResponse]
    total_nodes: int
    total_edges: int

class GraphSearchResponse(BaseModel):
    query: str
    matched_entities: list[EntityResponse]
    subgraph: KnowledgeGraphResponse
    execution_time_ms: float
