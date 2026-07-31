from typing import Any
from app.graph.base import BaseGraphStore
from app.schemas.entity import EntityResponse

class EntityReasoner:
    """
    Reasoning service for resolving entity co-occurrences, alias matching, and entity clustering.
    Operates on Graph Store without direct LLM API dependency.
    """
    def __init__(self, graph_store: BaseGraphStore) -> None:
        self.graph_store = graph_store

    async def resolve_aliases(self, entity_name: str) -> list[EntityResponse]:
        """Discovers entities sharing aliases or name variations."""
        matches = await self.graph_store.search_entities(query=entity_name, limit=10)
        return matches

    async def get_cooccurring_entities(self, entity_id: str) -> list[dict[str, Any]]:
        """Identifies entities frequently connected via relationship edges."""
        rels = await self.graph_store.get_neighbors(entity_id=entity_id, depth=1)
        co_occurring = []
        for r in rels:
            target_id = r.target_id if r.source_id == entity_id else r.source_id
            co_occurring.append({
                "entity_id": target_id,
                "relation_type": r.relation_type,
                "confidence": r.confidence
            })
        return co_occurring
