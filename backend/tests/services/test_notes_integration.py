"""
Unit tests for Notes Integration across SearchService, TimelineService, and AskService.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.mock_provider import MockLLMProvider
from app.services.search_service import SearchResult, SearchService, _compute_hybrid_boost
from app.services.timeline_service import (
    DailySummary,
    TimelineEvent,
    TimelineService,
    _classify_source,
)


def test_hybrid_boost_for_notes() -> None:
    query_tokens = ["rag", "architecture"]
    query_raw = "What notes do I have about RAG architecture?"
    rec = {"content": "Note Title: RAG Architecture Note\nTags: rag, memory"}
    meta = {
        "filepath": "notes/rag_arch.md",
        "title": "RAG Architecture Note",
        "tags": ["rag", "memory"],
        "headings": ["RAG Architecture"],
        "defined_symbols": ["rag", "memory", "RAG Architecture Note"],
    }
    boost = _compute_hybrid_boost(query_tokens, query_raw, rec, meta)
    assert boost >= 0.20


def test_classify_source_notes() -> None:
    res = SearchResult(
        id="note_01",
        filepath="notes/rag.md",
        filename="rag.md",
        start_line=1,
        end_line=10,
        score=0.90,
        content="Note about RAG",
        source="notes",
        meta_data={"title": "RAG Note"},
    )
    assert _classify_source(res) == "notes"


def test_timeline_event_notes_emoji() -> None:
    ev = TimelineEvent(
        timestamp=datetime.now(timezone.utc),
        source="notes",
        title="Note: RAG Architecture",
        summary="Note about RAG",
        filepath="notes/rag.md",
    )
    assert ev.source_emoji == "📝"


@pytest.mark.asyncio
async def test_timeline_service_format_notes() -> None:
    now = datetime.now(timezone.utc)
    ev = TimelineEvent(
        timestamp=now,
        source="notes",
        title="Note: RAG Memory",
        summary="Tags: rag, memory | Note about RAG Memory",
        filepath="notes/rag_memory.md",
    )
    svc = TimelineService(repo_path="/tmp/nonexistent")
    output = svc.format_events_as_timeline([ev], period_label="TODAY")
    assert "📝" in output
    assert "Note: RAG Memory" in output
    assert "notes=1" in output


@pytest.mark.asyncio
async def test_mock_llm_provider_notes_answer() -> None:
    llm = MockLLMProvider()
    res = SearchResult(
        id="note_01",
        filepath="notes/rag_system.md",
        filename="rag_system.md",
        start_line=1,
        end_line=20,
        score=0.95,
        content="Note Title: RAG System Architecture\nTags: rag, embeddings\n\n# RAG System\nDetailed notes on RAG.",
        source="notes",
        meta_data={
            "title": "RAG System Architecture",
            "tags": ["rag", "embeddings"],
        },
    )
    answer = await llm.generate_answer(
        query="What notes do I have about RAG?",
        context_prompt="...",
        results=[res],
    )

    assert "Matching notes found" in answer
    assert "RAG System Architecture" in answer
    assert "notes/rag_system.md" in answer
    assert "[Tags: rag, embeddings]" in answer
