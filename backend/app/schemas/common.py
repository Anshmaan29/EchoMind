from typing import Generic, TypeVar
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

class BaseResponseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class HealthResponse(BaseModel):
    status: str = Field(default="ok", json_schema_extra={"example": "ok"})

class PaginatedResponse(BaseResponseSchema, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
