from typing import Any
from app.graph.base import BaseGraphStore

class ProjectReasoner:
    """
    Reasoning service aggregating technology stacks, contributors, and project history.
    """
    def __init__(self, graph_store: BaseGraphStore) -> None:
        self.graph_store = graph_store

    async def get_project_tech_stack(self, project_name: str) -> dict[str, Any]:
        """Discovers frameworks, libraries, and technologies used by a project."""
        entities = await self.graph_store.search_entities(query=project_name, limit=5)
        if not entities:
            return {"project": project_name, "technologies": []}

        proj_id = entities[0].id
        rels = await self.graph_store.get_neighbors(entity_id=proj_id, depth=1)

        tech_stack = []
        for r in rels:
            if r.relation_type in ["USES", "DEPENDS_ON", "IMPLEMENTS"]:
                tech_stack.append({
                    "technology": r.target_id,
                    "relation": r.relation_type
                })

        return {
            "project": project_name,
            "project_id": proj_id,
            "technologies": tech_stack
        }
