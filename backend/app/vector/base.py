from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class VectorRecord(BaseModel):
    id: str
    vector: list[float]
    payload: dict[str, Any]

class VectorSearchResult(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]

class BaseVectorStore(ABC):
    """
    Abstract Base Class defining contract for Vector Stores in EchoMind.
    Enables future support for Pinecone, Milvus, Weaviate alongside Qdrant.
    """

    @abstractmethod
    async def initialize_collection(self, collection_name: str, dimension: int) -> None:
        """Initializes target collection/index with given vector dimension."""
        pass

    @abstractmethod
    async def upsert_records(self, collection_name: str, records: list[VectorRecord]) -> bool:
        """Upserts a list of vector records into the vector index."""
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0
    ) -> list[VectorSearchResult]:
        """Performs dense vector similarity search."""
        pass

    @abstractmethod
    async def delete_by_document_id(self, collection_name: str, document_id: str) -> bool:
        """Deletes all vector points associated with a document_id payload filter."""
        pass
