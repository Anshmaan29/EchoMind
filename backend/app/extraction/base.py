from abc import ABC, abstractmethod
from app.schemas.entity import EntityCreate
from app.schemas.relationship import RelationshipCreate

class BaseEntityExtractor(ABC):
    """Abstract Base Class for Entity Extraction Engines."""

    @abstractmethod
    async def extract_entities(self, text: str, source_document_id: str | None = None) -> list[EntityCreate]:
        """Extracts entity instances from raw or parsed document text."""
        pass

class BaseRelationshipExtractor(ABC):
    """Abstract Base Class for Relationship Extraction Engines."""

    @abstractmethod
    async def extract_relationships(
        self,
        text: str,
        entities: list[EntityCreate],
        source_document_id: str | None = None
    ) -> list[RelationshipCreate]:
        """Extracts relationship edges connecting extracted entities."""
        pass
