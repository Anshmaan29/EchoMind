from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class ChunkBase(BaseModel):
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    token_count: int = Field(default=0, ge=0)
    meta_data: dict[str, Any] = Field(default_factory=dict)

class ChunkCreate(ChunkBase):
    document_id: str
    vector_id: str | None = None

class ChunkResponse(ChunkBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    vector_id: str | None = None
    created_at: datetime
