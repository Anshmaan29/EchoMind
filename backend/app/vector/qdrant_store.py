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
    """
    def __init__(self, host: str = None, port: int = None, api_key: str = None) -> None:
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.api_key = api_key or settings.QDRANT_API_KEY
        
        try:
            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=10.0
            )
        except Exception as e:
            logger.warning(f"Could not connect to external Qdrant ({e}). Initializing in-memory Qdrant client.")
            self.client = AsyncQdrantClient(":memory:")

    async def initialize_collection(self, collection_name: str, dimension: int) -> None:
        try:
            collections = await self.client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            if collection_name not in existing_names:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection '{collection_name}' with dimension {dimension}.")
        except Exception as e:
            logger.warning(f"Failed to check/create Qdrant collection: {e}")

    async def upsert_records(self, collection_name: str, records: list[VectorRecord]) -> bool:
        if not records:
            return True

        points = [
            PointStruct(
                id=rec.id,
                vector=rec.vector,
                payload=rec.payload
            )
            for rec in records
        ]

        try:
            await self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Upserted {len(records)} points into Qdrant collection '{collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors into Qdrant: {e}")
            return False

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0
    ) -> list[VectorSearchResult]:
        try:
            if hasattr(self.client, "search"):
                hits = await self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold
                )
            elif hasattr(self.client, "query_points"):
                res = await self.client.query_points(
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
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> bool:
        try:
            await self.client.delete(
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
            logger.info(f"Deleted vector points for document_id '{document_id}'.")
            return True
        except Exception as e:
            logger.error(f"Qdrant point deletion error: {e}")
            return False
