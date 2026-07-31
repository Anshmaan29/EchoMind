import time
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.graph.factory import graph_store
from app.models.entity import Entity
from app.models.relationship import Relationship
from app.schemas.entity import EntityResponse
from app.schemas.graph import GraphSearchResponse, KnowledgeGraphResponse
from app.schemas.relationship import RelationshipResponse

class KnowledgeService:
    """
    Application Service managing Knowledge Graph entities, relationships, and subgraph searches.
    """
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session
        self.graph_store = graph_store

    async def list_entities(self, entity_type: str | None = None, limit: int = 50) -> list[EntityResponse]:
        stmt = select(Entity)
        if entity_type:
            stmt = stmt.where(Entity.type == entity_type)
        stmt = stmt.limit(limit)

        res = await self.db_session.execute(stmt)
        entities = res.scalars().all()
        return [EntityResponse.model_validate(e) for e in entities]

    async def get_entity_by_id(self, entity_id: str) -> EntityResponse | None:
        stmt = select(Entity).where(Entity.id == entity_id)
        res = await self.db_session.execute(stmt)
        entity = res.scalar_one_or_none()
        if entity:
            return EntityResponse.model_validate(entity)
        return await self.graph_store.find_entity(entity_id)

    async def list_relationships(self, limit: int = 50) -> list[RelationshipResponse]:
        stmt = select(Relationship).limit(limit)
        res = await self.db_session.execute(stmt)
        rels = res.scalars().all()
        return [RelationshipResponse.model_validate(r) for r in rels]

    async def search_graph(self, query: str, limit: int = 10) -> GraphSearchResponse:
        start = time.perf_counter()
        
        # 1. Search DB & Graph Store
        matched = await self.graph_store.search_entities(query=query, limit=limit)
        
        # 2. Get relationships
        all_rels: list[RelationshipResponse] = []
        for entity in matched:
            neighbors = await self.graph_store.get_neighbors(entity_id=entity.id, depth=1)
            all_rels.extend(neighbors)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return GraphSearchResponse(
            query=query,
            matched_entities=matched,
            subgraph=KnowledgeGraphResponse(
                entities=matched,
                relationships=all_rels,
                total_nodes=len(matched),
                total_edges=len(all_rels)
            ),
            execution_time_ms=elapsed_ms
        )

    async def get_neighbors(self, entity_id: str, depth: int = 1) -> KnowledgeGraphResponse:
        rels = await self.graph_store.get_neighbors(entity_id=entity_id, depth=depth)
        
        node_ids = set()
        node_ids.add(entity_id)
        for r in rels:
            node_ids.add(r.source_id)
            node_ids.add(r.target_id)

        entities: list[EntityResponse] = []
        for nid in node_ids:
            ent = await self.get_entity_by_id(nid)
            if ent:
                entities.append(ent)

        return KnowledgeGraphResponse(
            entities=entities,
            relationships=rels,
            total_nodes=len(entities),
            total_edges=len(rels)
        )
