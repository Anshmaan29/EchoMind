from typing import Any
from app.graph.base import BaseGraphStore

class RelationshipReasoner:
    """
    Reasoning service for multi-hop graph path analysis and dependency tracing.
    """
    def __init__(self, graph_store: BaseGraphStore) -> None:
        self.graph_store = graph_store

    async def trace_dependency_chain(self, start_entity_id: str) -> list[dict[str, Any]]:
        """Traces DEPENDS_ON and USES relationships originating from an entity."""
        rels = await self.graph_store.get_neighbors(entity_id=start_entity_id, depth=2)
        dependencies = []
        for r in rels:
            if r.relation_type in ["DEPENDS_ON", "USES", "IMPLEMENTS"]:
                dependencies.append({
                    "source": r.source_id,
                    "relation": r.relation_type,
                    "dependency": r.target_id,
                    "confidence": r.confidence
                })
        return dependencies
