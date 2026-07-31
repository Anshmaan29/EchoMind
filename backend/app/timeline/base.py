from abc import ABC, abstractmethod
from app.schemas.entity import EntityCreate
from app.schemas.timeline import TimelineEventCreate

class BaseTimelineEngine(ABC):
    """Abstract Base Class for Temporal Timeline Event Creation."""

    @abstractmethod
    async def create_timeline_events(
        self,
        text: str,
        entities: list[EntityCreate],
        source_document_id: str | None = None
    ) -> list[TimelineEventCreate]:
        """Converts text passages and entities into chronological Timeline Event objects."""
        pass
