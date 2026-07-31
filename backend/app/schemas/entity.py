from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

EntityType = Literal[
    "Person",
    "Organization",
    "Project",
    "Repository",
    "Technology",
    "Programming Language",
    "Framework",
    "Library",
    "API",
    "Document",
    "File",
    "URL",
    "Issue",
    "Pull Request",
    "Meeting",
    "Task",
    "Dataset",
    "Model",
    "Date",
    "Time",
    "Location",
    "Version",
    "Concept",
]

class EntityBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    type: EntityType = Field(...)
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    meta_data: dict[str, Any] = Field(default_factory=dict)

class EntityCreate(EntityBase):
    source_document_id: str | None = None

class EntityResponse(EntityBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_document_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
