import os
import pytest
from app.embeddings.backup import JSONLBackupWriter
from app.embeddings.checkpoint import CheckpointManager
from app.embeddings.mock_provider import MockEmbeddingProvider
from app.embeddings.pipeline import EmbeddingItem, GenericEmbeddingPipeline

@pytest.mark.asyncio
async def test_checkpoint_manager(tmp_path) -> None:
    db_file = str(tmp_path / "test_checkpoint.db")
    cp = CheckpointManager(db_path=db_file)

    assert not cp.is_processed("item_1")
    cp.mark_processed_batch(["item_1", "item_2"], source="pdf")
    assert cp.is_processed("item_1")
    assert cp.is_processed("item_2")
    assert len(cp.get_processed_ids()) == 2

    cp.reset()
    assert not cp.is_processed("item_1")

@pytest.mark.asyncio
async def test_jsonl_backup_writer(tmp_path) -> None:
    backup_file = str(tmp_path / "test_backup.jsonl")
    writer = JSONLBackupWriter(backup_filepath=backup_file)

    records = [
        {
            "id": "rec_1",
            "source": "pdf",
            "content": "Sample passage content",
            "embedding_model": "mock",
            "embedding_vector": [0.1, 0.2, 0.3],
            "metadata": {"doc_id": "d1"}
        }
    ]

    written = writer.write_records(records)
    assert written == 1
    assert os.path.exists(backup_file)

@pytest.mark.asyncio
async def test_generic_embedding_pipeline(tmp_path) -> None:
    cp_db = str(tmp_path / "cp.db")
    bk_file = str(tmp_path / "backup.jsonl")

    pipeline = GenericEmbeddingPipeline(
        embedder=MockEmbeddingProvider(dimension=32),
        checkpoint_db_path=cp_db,
        backup_filepath=bk_file
    )

    items = [
        EmbeddingItem(id="item_10", source="pdf", content="Passage 1"),
        EmbeddingItem(id="item_11", source="github", content="Code snippet 1"),
        EmbeddingItem(id="item_12", source="timeline", content="Milestone 1 event"),
    ]

    # First run
    metrics = await pipeline.process_items(items=items, batch_size=2, max_workers=2, resume=True)
    assert metrics.processed_items == 3
    assert metrics.skipped_items == 0
    assert metrics.embeddings_per_sec >= 0.0

    # Second run with resume (should skip all 3 items)
    metrics_resumed = await pipeline.process_items(items=items, batch_size=2, max_workers=2, resume=True)
    assert metrics_resumed.processed_items == 0
    assert metrics_resumed.skipped_items == 3
