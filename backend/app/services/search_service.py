import json
import os
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingProvider
from app.embeddings.factory import embedding_provider
from app.vector.base import BaseVectorStore
from app.vector.factory import vector_store

class SearchResult(BaseModel):
    """
    Standardized Hybrid Search Result container preserving file paths and line numbers.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    filepath: str
    filename: str
    start_line: int
    end_line: int
    score: float
    content: str
    source: str
    meta_data: dict[str, Any] = Field(default_factory=dict)

class SearchService:
    """
    Provider-Agnostic Hybrid Vector Search Service.
    Queries Qdrant vector store when available and falls back to JSONL vector backups
    (embeddings_backup.jsonl) using NumPy cosine similarity.
    """
    def __init__(
        self,
        embedder: BaseEmbeddingProvider = None,
        vector_store_inst: BaseVectorStore = None,
        backup_filepath: str = None
    ) -> None:
        self.embedder = embedder or embedding_provider
        self.vector_store = vector_store_inst or vector_store
        self.backup_filepath = backup_filepath or self._discover_backup_filepath()

    def _discover_backup_filepath(self) -> str:
        """Discovers available embeddings_backup.jsonl file across workspace paths."""
        candidates = [
            "data/embeddings_backup.jsonl",
            "../data/embeddings_backup.jsonl",
            "embeddings_backup.jsonl",
            "../embeddings_backup.jsonl",
            "experiments/outputs/embeddings_backup.jsonl",
            "/Users/anshmaansingh/Echomind/data/embeddings_backup.jsonl",
            "/Users/anshmaansingh/Echomind/backend/data/embeddings_backup.jsonl",
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 0:
                return c
        return "data/embeddings_backup.jsonl"

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        collection_name: str = None
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        target_collection = collection_name or settings.QDRANT_COLLECTION_NAME
        query_vec = await self.embedder.embed_single(query)

        # 1. Try Qdrant Vector Store Search
        qdrant_results: list[SearchResult] = []
        try:
            hits = await self.vector_store.search(
                collection_name=target_collection,
                query_vector=query_vec,
                limit=top_k,
                score_threshold=min_score
            )
            for hit in hits:
                payload = hit.payload or {}
                qdrant_results.append(
                    SearchResult(
                        id=hit.id,
                        filepath=payload.get("filepath", payload.get("filename", "unknown")),
                        filename=payload.get("filename", "unknown"),
                        start_line=payload.get("start_line", 1),
                        end_line=payload.get("end_line", 1),
                        score=round(hit.score, 4),
                        content=payload.get("content", ""),
                        source=payload.get("source", "unknown"),
                        meta_data=payload
                    )
                )
        except Exception as e:
            logger.warning(f"Qdrant search unfulfilled ({e}). Falling back to JSONL backup file search.")

        if qdrant_results:
            return qdrant_results

        # 2. Fallback to Local JSONL Vector Backup Search
        return self._search_jsonl_backup(query_vec=query_vec, top_k=top_k, min_score=min_score)

    def _search_jsonl_backup(
        self,
        query_vec: list[float],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        if not os.path.exists(self.backup_filepath):
            logger.warning(f"JSONL backup file '{self.backup_filepath}' not found.")
            return []

        records: list[dict[str, Any]] = []
        vectors: list[list[float]] = []

        try:
            with open(self.backup_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    vec = rec.get("embedding_vector")
                    if vec and len(vec) == len(query_vec):
                        records.append(rec)
                        vectors.append(vec)
        except Exception as e:
            logger.error(f"Error reading JSONL backup '{self.backup_filepath}': {e}")
            return []

        if not vectors:
            return []

        # NumPy Cosine Similarity
        matrix = np.array(vectors, dtype=np.float32)
        q_vec = np.array(query_vec, dtype=np.float32)

        norm_q = np.linalg.norm(q_vec)
        if norm_q == 0:
            norm_q = 1e-9

        norm_matrix = np.linalg.norm(matrix, axis=1)
        norm_matrix[norm_matrix == 0] = 1e-9

        similarities = np.dot(matrix, q_vec) / (norm_matrix * norm_q)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1]

        results: list[SearchResult] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue

            rec = records[idx]
            meta = rec.get("metadata", {})
            results.append(
                SearchResult(
                    id=str(rec.get("id", f"backup_{idx}")),
                    filepath=str(meta.get("filepath", meta.get("filename", "unknown"))),
                    filename=str(meta.get("filename", "unknown")),
                    start_line=int(meta.get("start_line", 1)),
                    end_line=int(meta.get("end_line", 1)),
                    score=round(score, 4),
                    content=str(rec.get("content", "")),
                    source=str(rec.get("source", "unknown")),
                    meta_data=meta
                )
            )

            if len(results) >= top_k:
                break

        return results
