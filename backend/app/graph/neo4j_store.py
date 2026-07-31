from datetime import datetime, timezone
from typing import Any
from neo4j import AsyncGraphDatabase
from app.core.config import settings
from app.core.logging import logger
from app.graph.base import BaseGraphStore
from app.schemas.entity import EntityResponse
from app.schemas.relationship import RelationshipResponse

class Neo4jGraphStore(BaseGraphStore):
    """
    Neo4j Graph Store Implementation using official AsyncGraphDatabase driver.
    Includes in-memory fallback dictionary storage when live Neo4j is unavailable.
    """
    def __init__(self, uri: str = None, user: str = None, password: str = None) -> None:
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.database = settings.NEO4J_DATABASE
        self.driver = None
        self._is_standalone_fallback = False

        # In-memory fallback dictionary tables
        self._mem_entities: dict[str, EntityResponse] = {}
        self._mem_relationships: list[RelationshipResponse] = []

    async def initialize(self) -> None:
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=5.0
            )
            # Verify connectivity
            async with self.driver.session(database=self.database) as session:
                await session.run("RETURN 1")
            logger.info(f"Connected successfully to Neo4j database at {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j instance unavailable at {self.uri} ({e}). Falling back to in-memory graph engine.")
            self._is_standalone_fallback = True

    async def create_entity(self, entity: EntityResponse) -> EntityResponse:
        self._mem_entities[entity.id] = entity

        if self._is_standalone_fallback or not self.driver:
            return entity

        cypher = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.aliases = $aliases,
            e.description = $description,
            e.confidence = $confidence,
            e.source_document_id = $source_document_id,
            e.created_at = $created_at
        RETURN e
        """
        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(
                    cypher,
                    id=entity.id,
                    name=entity.name,
                    type=entity.type,
                    aliases=entity.aliases,
                    description=entity.description or "",
                    confidence=entity.confidence,
                    source_document_id=entity.source_document_id or "",
                    created_at=entity.created_at.isoformat()
                )
        except Exception as e:
            logger.error(f"Neo4j Cypher create_entity error: {e}")

        return entity

    async def create_relationship(self, relationship: RelationshipResponse) -> RelationshipResponse:
        self._mem_relationships.append(relationship)

        if self._is_standalone_fallback or not self.driver:
            return relationship

        # Sanitize relationship type for Cypher
        rel_type = relationship.relation_type.upper().replace(" ", "_")
        cypher = f"""
        MATCH (s:Entity {{id: $source_id}})
        MATCH (t:Entity {{id: $target_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r.id = $id,
            r.confidence = $confidence,
            r.evidence = $evidence,
            r.source_document_id = $source_document_id,
            r.timestamp = $timestamp
        RETURN r
        """
        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(
                    cypher,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    id=relationship.id,
                    confidence=relationship.confidence,
                    evidence=relationship.evidence or "",
                    source_document_id=relationship.source_document_id or "",
                    timestamp=relationship.timestamp.isoformat()
                )
        except Exception as e:
            logger.error(f"Neo4j Cypher create_relationship error: {e}")

        return relationship

    async def update_entity(self, entity_id: str, properties: dict[str, Any]) -> EntityResponse | None:
        if entity_id in self._mem_entities:
            existing = self._mem_entities[entity_id].model_dump()
            existing.update(properties)
            updated = EntityResponse.model_validate(existing)
            self._mem_entities[entity_id] = updated
            return updated
        return None

    async def find_entity(self, entity_id: str) -> EntityResponse | None:
        return self._mem_entities.get(entity_id)

    async def search_entities(self, query: str, entity_type: str | None = None, limit: int = 10) -> list[EntityResponse]:
        query_lower = query.lower()
        results = []

        for entity in self._mem_entities.values():
            if entity_type and entity.type.lower() != entity_type.lower():
                continue
            if query_lower in entity.name.lower() or (entity.description and query_lower in entity.description.lower()):
                results.append(entity)
            if len(results) >= limit:
                break

        return results

    async def get_neighbors(self, entity_id: str, depth: int = 1) -> list[RelationshipResponse]:
        matching_rels = [
            rel for rel in self._mem_relationships
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
        return matching_rels

    async def delete_entity(self, entity_id: str) -> bool:
        if entity_id in self._mem_entities:
            del self._mem_entities[entity_id]
            self._mem_relationships = [
                rel for rel in self._mem_relationships
                if rel.source_id != entity_id and rel.target_id != entity_id
            ]
            return True
        return False

    async def query_graph(self, cypher_or_query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._is_standalone_fallback or not self.driver:
            return [{"status": "fallback", "nodes_count": len(self._mem_entities)}]

        try:
            async with self.driver.session(database=self.database) as session:
                res = await session.run(cypher_or_query, params or {})
                records = await res.data()
                return records
        except Exception as e:
            logger.error(f"Neo4j Cypher query error: {e}")
            return []
