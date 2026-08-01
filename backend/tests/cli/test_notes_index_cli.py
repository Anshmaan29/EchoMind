"""
Unit tests for app.cli.notes_index — Phase 3.0 Personal Notes CLI Indexer.
"""
from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.cli.notes_index import main_async
from app.embeddings.pipeline import EmbeddingItem, PipelineMetrics


@pytest.mark.asyncio
async def test_notes_index_cli_execution(tmp_path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    note_file = notes_dir / "sample.md"
    note_file.write_text("# Sample Note\nContent here.", encoding="utf-8")

    mock_metrics = PipelineMetrics(
        total_items=1,
        processed_items=1,
        skipped_items=0,
        failed_items=0,
        elapsed_seconds=0.1,
    )

    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.process_items = AsyncMock(return_value=mock_metrics)

    args = argparse.Namespace(
        notes_dir=str(notes_dir),
        provider=None,
        batch_size=64,
        workers=4,
        collection_name="test_collection",
        backup=None,
        no_resume=False,
    )

    with patch("app.cli.notes_index.GenericEmbeddingPipeline", return_value=mock_pipeline_inst):
        await main_async(args)

    mock_pipeline_inst.process_items.assert_called_once()
    items = mock_pipeline_inst.process_items.call_args.kwargs["items"]
    assert len(items) == 1
    assert items[0].source == "notes"
    assert items[0].meta_data["filename"] == "sample.md"


@pytest.mark.asyncio
async def test_notes_index_cli_no_notes(tmp_path, capsys) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    args = argparse.Namespace(
        notes_dir=str(empty_dir),
        provider=None,
        batch_size=64,
        workers=4,
        collection_name="test_collection",
        backup=None,
        no_resume=False,
    )

    await main_async(args)
    captured = capsys.readouterr()
    assert "No notes (.md, .txt) found to index" in captured.out
