import asyncio
import time
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.backup import JSONLBackupWriter
from app.embeddings.base import BaseEmbeddingProvider
from app.embeddings.checkpoint import CheckpointManager
from app.embeddings.factory import embedding_provider
from app.vector.base import BaseVectorStore, VectorRecord
from app.vector.factory import vector_store

SourceType = Literal["pdf", "image", "github", "timeline", "entity"]

class EmbeddingItem(BaseModel):
    """
    Standardized payload schema for embedding generation across all digital memory sources.
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    source: SourceType
    content: str = Field(..., min_length=1)
    meta_data: dict[str, Any] = Field(default_factory=dict)

class PipelineMetrics(BaseModel):
    total_items: int = 0
    processed_items: int = 0
    skipped_items: int = 0
    failed_items: int = 0
    elapsed_seconds: float = 0.0
    embeddings_per_sec: float = 0.0
    documents_per_sec: float = 0.0

class GenericEmbeddingPipeline:
    """
    Production-Grade Batching & Async Embedding Generation Pipeline.
    Supports PDF chunks, Image OCR/captions, GitHub files, Timeline events, and Knowledge Graph entities.
    Optimized for high GPU throughput (A100), automated failure resumption, Qdrant indexing, and local JSONL backups.
    """
    def __init__(
        self,
        embedder: BaseEmbeddingProvider = None,
        vector_store_inst: BaseVectorStore = None,
        checkpoint_db_path: str = ".checkpoints/embedding_checkpoint.db",
        backup_filepath: str = "data/embeddings_backup.jsonl"
    ) -> None:
        self.embedder = embedder or embedding_provider
        self.vector_store = vector_store_inst or vector_store
        self.checkpoint = CheckpointManager(db_path=checkpoint_db_path)
        self.backup_writer = JSONLBackupWriter(backup_filepath=backup_filepath)

    async def process_items(
        self,
        items: list[EmbeddingItem],
        collection_name: str = None,
        batch_size: int = 64,
        max_workers: int = 4,
        resume: bool = True,
        max_retries: int = 3
    ) -> PipelineMetrics:
        start_time = time.perf_counter()
        target_collection = collection_name or settings.QDRANT_COLLECTION_NAME

        # Ensure vector collection exists
        await self.vector_store.initialize_collection(
            collection_name=target_collection,
            dimension=self.embedder.dimension
        )

        # Filter out already processed items if resumption is enabled
        if resume:
            processed_set = self.checkpoint.get_processed_ids()
            pending_items = [it for it in items if it.id not in processed_set]
            skipped_count = len(items) - len(pending_items)
        else:
            pending_items = items
            skipped_count = 0

        logger.info(
            f"Starting EmbeddingPipeline processing: {len(pending_items)} items pending "
            f"({skipped_count} skipped via checkpoint). Batch size: {batch_size}, Workers: {max_workers}."
        )

        if not pending_items:
            elapsed = round(time.perf_counter() - start_time, 2)
            return PipelineMetrics(
                total_items=len(items),
                processed_items=0,
                skipped_items=skipped_count,
                failed_items=0,
                elapsed_seconds=elapsed,
                embeddings_per_sec=0.0,
                documents_per_sec=0.0
            )

        # Segment pending items into batches for high GPU utilization
        batches = [
            pending_items[i : i + batch_size]
            for i in range(0, len(pending_items), batch_size)
        ]

        semaphore = asyncio.Semaphore(max_workers)
        processed_count = 0
        failed_count = 0

        async def process_single_batch(batch: list[EmbeddingItem]) -> int:
            nonlocal processed_count, failed_count
            async with semaphore:
                texts = [b.content for b in batch]
                
                # Retry logic for embedding generation
                embeddings = None
                for attempt in range(1, max_retries + 1):
                    try:
                        embeddings = await self.embedder.embed_texts(texts)
                        break
                    except Exception as e:
                        logger.warning(f"Embedding batch attempt {attempt}/{max_retries} failed: {e}")
                        if attempt == max_retries:
                            logger.error(f"Failed to generate embeddings for batch of {len(batch)} items after {max_retries} retries.")
                            failed_count += len(batch)
                            return 0
                        await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

                if not embeddings or len(embeddings) != len(batch):
                    failed_count += len(batch)
                    return 0

                # Build Vector Records & Backup Payloads
                vector_records: list[VectorRecord] = []
                backup_records: list[dict[str, Any]] = []
                batch_ids: list[str] = []

                for item, emb_vec in zip(batch, embeddings):
                    vector_records.append(
                        VectorRecord(
                            id=item.id,
                            vector=emb_vec,
                            payload={
                                "source": item.source,
                                "content": item.content,
                                **item.meta_data
                            }
                        )
                    )
                    backup_records.append({
                        "id": item.id,
                        "source": item.source,
                        "content": item.content,
                        "embedding_model": getattr(self.embedder, "model_name", settings.EMBEDDING_MODEL_NAME),
                        "embedding_vector": emb_vec,
                        "metadata": item.meta_data
                    })
                    batch_ids.append(item.id)

                # Upsert to Qdrant Vector Store
                await self.vector_store.upsert_records(
                    collection_name=target_collection,
                    records=vector_records
                )

                # Write Local JSONL Backup
                self.backup_writer.write_records(backup_records)

                # Checkpoint progress
                self.checkpoint.mark_processed_batch(
                    item_ids=batch_ids,
                    source=batch[0].source if batch else "unknown"
                )

                processed_count += len(batch)
                return len(batch)

        # Run worker tasks concurrently
        tasks = [process_single_batch(b) for b in batches]
        await asyncio.gather(*tasks)

        elapsed = max(0.001, round(time.perf_counter() - start_time, 2))
        embeddings_per_sec = round(processed_count / elapsed, 2)
        documents_per_sec = round((processed_count + skipped_count) / elapsed, 2)

        metrics = PipelineMetrics(
            total_items=len(items),
            processed_items=processed_count,
            skipped_items=skipped_count,
            failed_items=failed_count,
            elapsed_seconds=elapsed,
            embeddings_per_sec=embeddings_per_sec,
            documents_per_sec=documents_per_sec
        )

        logger.info(
            f"Pipeline Execution Complete: Processed={processed_count}, Skipped={skipped_count}, "
            f"Failed={failed_count} in {elapsed}s ({embeddings_per_sec} embeddings/sec)."
        )

        return metrics
