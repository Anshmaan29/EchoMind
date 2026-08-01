"""
Unit tests for NoteConnector — Phase 3.0 Personal Notes Memory.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
import pytest

from app.embeddings.pipeline import EmbeddingItem
from app.ingestion.note_connector import (
    NoteConnector,
    extract_file_dates,
    extract_headings,
    extract_links,
    extract_tags,
    extract_title,
    parse_frontmatter,
)


def test_parse_frontmatter_valid() -> None:
    raw = """---
title: "Architecture Design"
tags: [rag, memory, architecture]
created: 2026-07-28
---
# Body Title
Some body text.
"""
    fm, body = parse_frontmatter(raw)
    assert fm["title"] == "Architecture Design"
    assert fm["tags"] == ["rag", "memory", "architecture"]
    assert fm["created"] == "2026-07-28"
    assert "# Body Title" in body


def test_parse_frontmatter_none() -> None:
    raw = "# Just a Note\nNo frontmatter here."
    fm, body = parse_frontmatter(raw)
    assert fm == {}
    assert body == raw


def test_extract_title_frontmatter() -> None:
    fm = {"title": "Explicit Title"}
    body = "# Heading Title\nBody text"
    assert extract_title(fm, body, "note.md") == "Explicit Title"


def test_extract_title_h1_header() -> None:
    fm = {}
    body = "# Heading Title\nBody text"
    assert extract_title(fm, body, "my-note.md") == "Heading Title"


def test_extract_title_filename_fallback() -> None:
    fm = {}
    body = "No heading here, just body text."
    assert extract_title(fm, body, "my_architecture_note.md") == "My Architecture Note"


def test_extract_headings() -> None:
    text = """# Main Header
Intro text.
## Section 1: Overview
### Subsection 1.1
## Section 2: Conclusion
"""
    headings = extract_headings(text)
    assert headings == [
        "Main Header",
        "Section 1: Overview",
        "Subsection 1.1",
        "Section 2: Conclusion",
    ]


def test_extract_tags() -> None:
    fm = {"tags": ["rag", "embeddings"]}
    body = "Text with #memory and #architecture hashtags. Also #rag."
    tags = extract_tags(fm, body)
    assert "rag" in tags
    assert "embeddings" in tags
    assert "memory" in tags
    assert "architecture" in tags


def test_extract_links() -> None:
    text = """
Check [RAG Docs](docs/rag.md) and [External](https://google.com).
Also see [[Memory Architecture]] and [[embeddings|Embeddings Note]].
"""
    links = extract_links(text)
    assert "docs/rag.md" in links
    assert "Memory Architecture" in links
    assert "embeddings" in links
    assert "https://google.com" not in links  # external URLs ignored


def test_extract_file_dates(tmp_path) -> None:
    note_file = tmp_path / "test.md"
    note_file.write_text("# Test Note", encoding="utf-8")
    fm = {"created": "2026-07-01T00:00:00Z", "modified": "2026-08-01T00:00:00Z"}
    created, modified = extract_file_dates(str(note_file), fm)
    assert created == "2026-07-01T00:00:00Z"
    assert modified == "2026-08-01T00:00:00Z"


def test_note_connector_scan_notes(tmp_path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    sub_dir = notes_dir / "architecture"
    sub_dir.mkdir()

    note1 = notes_dir / "rag_overview.md"
    note1.write_text(
        """---
title: RAG Memory System
tags: [rag, embeddings]
---
# RAG Memory Overview
This is a note about RAG and vector search #vector.
See [[Architecture]] for details.
""",
        encoding="utf-8",
    )

    note2 = sub_dir / "ideas.txt"
    note2.write_text("Idea 1: Notes ingestion pipeline.\nIdea 2: Embeddings.", encoding="utf-8")

    connector = NoteConnector(notes_dir=str(notes_dir))
    items = connector.scan_notes()

    assert len(items) == 2
    for item in items:
        assert isinstance(item, EmbeddingItem)
        assert item.source == "notes"
        assert "filepath" in item.meta_data
        assert "title" in item.meta_data
        assert "tags" in item.meta_data
        assert "headings" in item.meta_data
        assert "links" in item.meta_data
        assert "created_date" in item.meta_data
        assert "modified_date" in item.meta_data

    # Check note1 specific metadata
    rag_item = [i for i in items if i.meta_data["filename"] == "rag_overview.md"][0]
    assert rag_item.meta_data["title"] == "RAG Memory System"
    assert "rag" in rag_item.meta_data["tags"]
    assert "vector" in rag_item.meta_data["tags"]
    assert "RAG Memory Overview" in rag_item.meta_data["headings"]
    assert "Architecture" in rag_item.meta_data["links"]


def test_discover_notes_dir(tmp_path) -> None:
    from app.ingestion.note_connector import discover_notes_dir

    custom_dir = tmp_path / "my_custom_notes"
    custom_dir.mkdir()
    assert discover_notes_dir(str(custom_dir)) == str(custom_dir.resolve())

    # When None is passed, auto-discovery returns an absolute path ending in notes
    discovered = discover_notes_dir(None)
    assert isinstance(discovered, str)
    assert len(discovered) > 0

