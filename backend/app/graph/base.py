from abc import ABC, abstractmethod
from typing import Any
from app.schemas.entity import EntityResponse
from app.schemas.relationship import RelationshipResponse

class BaseGraphStore(ABC):
    """
    Abstract Base Class contract for Knowledge Graph stores in EchoMind.
    Exposes methods for entity and relationship persistence, neighbor discovery, and graph search.
    Supports Neo4j, Memgraph, and Amazon Neptune backends.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initializes database connections, constraints, and indexes."""
        pass

    @abstractmethod
    async def create_entity(self, entity: EntityResponse) -> EntityResponse:
        """Creates or merges an entity node in the knowledge graph."""
        pass

    @abstractmethod
    async def create_relationship(self, relationship: RelationshipResponse) -> RelationshipResponse:
        """Creates a directed relationship edge between two entities."""
        pass

    @abstractmethod
    async def update_entity(self, entity_id: str, properties: dict[str, Any]) -> EntityResponse | None:
        """Updates properties of an existing entity node."""
        pass

    @abstractmethod
    async def find_entity(self, entity_id: str) -> EntityResponse | None:
        """Finds an entity node by its unique ID."""
        pass

    @abstractmethod
    async def search_entities(self, query: str, entity_type: str | None = None, limit: int = 10) -> list[EntityResponse]:
        """Searches entity nodes by name or description matching."""
        pass

    @abstractmethod
    async def get_neighbors(self, entity_id: str, depth: int = 1) -> list[RelationshipResponse]:
        """Retrieves relationship edges connected to an entity up to N-hops depth."""
        pass

    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """Deletes an entity node and its connected relationship edges."""
        pass

    @abstractmethod
    async def query_graph(self, cypher_or_query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Executes a native graph query."""
        pass
