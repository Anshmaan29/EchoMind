from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class TimelineEventBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entities_involved: list[str] = Field(default_factory=list)
    projects_involved: list[str] = Field(default_factory=list)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    meta_data: dict[str, Any] = Field(default_factory=dict)

class TimelineEventCreate(TimelineEventBase):
    source_document_id: str | None = None

class TimelineEventResponse(TimelineEventBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_document_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TimelineResponse(BaseModel):
    project_name: str | None = None
    events: list[TimelineEventResponse]
    total_events: int
