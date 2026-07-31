from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

RelationType = Literal[
    "CREATED",
    "USES",
    "DEPENDS_ON",
    "PART_OF",
    "OWNS",
    "WORKED_ON",
    "RELATED_TO",
    "MENTIONS",
    "REFERENCES",
    "GENERATES",
    "IMPLEMENTS",
    "FIXES",
    "DISCUSSED_IN",
    "LOCATED_IN",
    "TRAINED_ON",
    "CONNECTED_TO",
]

class RelationshipBase(BaseModel):
    source_id: str = Field(...)
    target_id: str = Field(...)
    relation_type: RelationType = Field(...)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str | None = None
    meta_data: dict[str, Any] = Field(default_factory=dict)

class RelationshipCreate(RelationshipBase):
    source_document_id: str | None = None

class RelationshipResponse(RelationshipBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_document_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
