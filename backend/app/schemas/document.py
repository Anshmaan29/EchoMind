from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.chunk import ChunkResponse

class DocumentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    source_type: str = Field(default="pdf", max_length=50)
    meta_data: dict[str, Any] = Field(default_factory=dict)

class DocumentCreate(DocumentBase):
    uri_or_path: str | None = None
    file_hash: str | None = None
    file_size_bytes: int = 0
    raw_text: str | None = None

class DocumentResponse(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    file_size_bytes: int
    created_at: datetime
    updated_at: datetime

class DocumentDetailResponse(DocumentResponse):
    chunks: list[ChunkResponse] = Field(default_factory=list)
