from typing import Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from app.core.config import settings
from app.core.logging import logger
from app.vector.base import BaseVectorStore, VectorRecord, VectorSearchResult

class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant Vector Store Implementation supporting REST and gRPC operations.
    Features single-warning connection tracking for clean local development without Qdrant.
    """
    def __init__(self, host: str = None, port: int = None, api_key: str = None) -> None:
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.is_available: bool | None = None
        self._client: AsyncQdrantClient | None = None

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=2.0
            )
        return self._client

    async def _check_availability(self) -> bool:
        if self.is_available is not None:
            return self.is_available

        try:
            client = self._get_client()
            await client.get_collections()
            self.is_available = True
            return True
        except Exception:
            self.is_available = False
            logger.warning("Qdrant unavailable. Using local JSONL backup.")
            return False

    async def initialize_collection(self, collection_name: str, dimension: int) -> None:
        if not await self._check_availability():
            return

        try:
            client = self._get_client()
            collections = await client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            if collection_name not in existing_names:
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection '{collection_name}' with dimension {dimension}.")
        except Exception as e:
            self.is_available = False
            logger.warning("Qdrant unavailable. Using local JSONL backup.")

    async def upsert_records(self, collection_name: str, records: list[VectorRecord]) -> bool:
        if not records:
            return True

        if not await self._check_availability():
            return False

        points = [
            PointStruct(
                id=rec.id,
                vector=rec.vector,
                payload=rec.payload
            )
            for rec in records
        ]

        try:
            client = self._get_client()
            await client.upsert(collection_name=collection_name, points=points)
            return True
        except Exception:
            self.is_available = False
            logger.warning("Qdrant unavailable. Using local JSONL backup.")
            return False

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0
    ) -> list[VectorSearchResult]:
        if not await self._check_availability():
            return []

        try:
            client = self._get_client()
            if hasattr(client, "search"):
                hits = await client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold
                )
            elif hasattr(client, "query_points"):
                res = await client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit,
                    score_threshold=score_threshold
                )
                hits = res.points
            else:
                hits = []

            return [
                VectorSearchResult(
                    id=str(hit.id),
                    score=float(hit.score),
                    payload=hit.payload or {}
                )
                for hit in hits
            ]
        except Exception:
            self.is_available = False
            logger.warning("Qdrant unavailable. Using local JSONL backup.")
            return []

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> bool:
        if not await self._check_availability():
            return False

        try:
            client = self._get_client()
            await client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            return True
        except Exception:
            self.is_available = False
            return False
