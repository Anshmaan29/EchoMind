"""
EchoMind Personal Notes Memory Connector — Phase 3.0

Recursively scans a directory of Markdown (.md) and text (.txt) notes,
extracts structured metadata (title, headings, tags, links, created date,
modified date, content), and generates line-annotated EmbeddingItems with
source="notes" for the embedding pipeline.
"""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.logging import logger
from app.embeddings.pipeline import EmbeddingItem

# Regex patterns for note parsing
_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n\s*---\s*\n", re.DOTALL)
_H1_HEADER_RE = re.compile(r"^\s*#\s+(.+)$", re.MULTILINE)
_ALL_HEADINGS_RE = re.compile(r"^\s*(#{1,6})\s+(.+)$", re.MULTILINE)
_HASHTAG_RE = re.compile(r"(?<!\S)#([a-zA-Z0-9_-]+)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """
    Extracts YAML frontmatter key-value pairs from the top of markdown text.
    Returns (frontmatter_dict, body_text).
    """
    match = _FRONTMATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text

    fm_block = match.group(1)
    body_text = raw_text[match.end():]
    fm_dict: dict[str, Any] = {}

    for line in fm_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower()
        val = val.strip()

        if val.startswith("[") and val.endswith("]"):
            val_list = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
            fm_dict[key] = val_list
        elif "," in val and not val.startswith(("'", '"')):
            val_list = [v.strip().strip("'\"") for v in val.split(",") if v.strip()]
            fm_dict[key] = val_list
        else:
            fm_dict[key] = val.strip("'\"")

    return fm_dict, body_text


def extract_title(fm_dict: dict[str, Any], body_text: str, fallback_filename: str) -> str:
    """
    Extracts title from frontmatter, first H1 header, or filename fallback.
    """
    if "title" in fm_dict and str(fm_dict["title"]).strip():
        return str(fm_dict["title"]).strip()

    h1_match = _H1_HEADER_RE.search(body_text)
    if h1_match:
        return h1_match.group(1).strip()

    stem = os.path.splitext(fallback_filename)[0]
    return stem.replace("-", " ").replace("_", " ").title()


def extract_headings(text: str) -> list[str]:
    """Extracts list of all heading titles in the markdown text."""
    headings: list[str] = []
    for match in _ALL_HEADINGS_RE.finditer(text):
        title = match.group(2).strip()
        if title and title not in headings:
            headings.append(title)
    return headings


def extract_tags(fm_dict: dict[str, Any], body_text: str) -> list[str]:
    """Extracts tags from frontmatter and inline hashtags."""
    tags: set[str] = set()

    fm_tags = fm_dict.get("tags") or fm_dict.get("tag")
    if isinstance(fm_tags, list):
        for t in fm_tags:
            t_str = str(t).strip().lstrip("#").lower()
            if t_str:
                tags.add(t_str)
    elif isinstance(fm_tags, str):
        for t in fm_tags.split(","):
            t_str = t.strip().lstrip("#").lower()
            if t_str:
                tags.add(t_str)

    for match in _HASHTAG_RE.finditer(body_text):
        tag = match.group(1).lower()
        if not tag.isdigit() and len(tag) >= 2:
            tags.add(tag)

    return sorted(list(tags))


def extract_links(text: str) -> list[str]:
    """Extracts markdown links [text](target) and wiki links [[target]]."""
    links: set[str] = set()

    for match in _MARKDOWN_LINK_RE.finditer(text):
        target = match.group(2).strip()
        if target and not target.startswith(("http://", "https://", "mailto:")):
            links.add(target)

    for match in _WIKI_LINK_RE.finditer(text):
        target = match.group(1).strip()
        if target:
            links.add(target)

    return sorted(list(links))


def extract_file_dates(filepath: str, fm_dict: dict[str, Any]) -> tuple[str, str]:
    """
    Extracts created and modified ISO dates from frontmatter or file stat.
    Returns (created_date_iso, modified_date_iso).
    """
    created_iso = ""
    modified_iso = ""

    for k in ("created", "date", "created_at"):
        if k in fm_dict and fm_dict[k]:
            created_iso = str(fm_dict[k]).strip()
            break

    for k in ("modified", "updated", "updated_at"):
        if k in fm_dict and fm_dict[k]:
            modified_iso = str(fm_dict[k]).strip()
            break

    try:
        stat = os.stat(filepath)
        if not modified_iso:
            modified_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        if not created_iso:
            birthtime = getattr(stat, "st_birthtime", stat.st_ctime)
            created_iso = datetime.fromtimestamp(birthtime, tz=timezone.utc).isoformat()
    except Exception:
        now_iso = datetime.now(timezone.utc).isoformat()
        created_iso = created_iso or now_iso
        modified_iso = modified_iso or now_iso

    return created_iso, modified_iso


def discover_notes_dir(notes_dir: str | None = None) -> str:
    """
    Discovers project notes directory automatically.
    If notes_dir is explicitly provided and exists, uses it.
    Otherwise searches for project root (.git) or existing notes/ directories.
    """
    if notes_dir:
        abs_p = os.path.abspath(notes_dir)
        if os.path.exists(abs_p):
            return abs_p

    candidates = [
        "../notes",
        "notes",
        "../../notes",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../notes")),
    ]
    for c in candidates:
        abs_c = os.path.abspath(c)
        if os.path.exists(abs_c) and os.path.isdir(abs_c):
            return abs_c

    # Walk up to find .git directory
    curr = os.path.abspath(".")
    while True:
        if os.path.isdir(os.path.join(curr, ".git")):
            target = os.path.join(curr, "notes")
            return os.path.abspath(target)
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    return os.path.abspath(notes_dir or "../notes")


class NoteConnector:
    """
    Personal Notes Connector scanning Markdown (.md) and text (.txt) files.
    """
    SUPPORTED_EXTENSIONS = {".md", ".txt"}

    def __init__(self, notes_dir: str | None = None) -> None:
        self.notes_dir = discover_notes_dir(notes_dir)

    def process_file(self, filepath: str, root_dir: str) -> list[EmbeddingItem]:
        """Reads a single note file and converts it into EmbeddingItem chunks."""
        items: list[EmbeddingItem] = []
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        rel_path = os.path.relpath(filepath, root_dir)

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except Exception as e:
            logger.warning(f"Could not read note file '{filepath}': {e}")
            return items

        if not raw_text.strip():
            return items

        fm_dict, body_text = parse_frontmatter(raw_text)
        title = extract_title(fm_dict, body_text, filename)
        headings = extract_headings(body_text)
        tags = extract_tags(fm_dict, body_text)
        links = extract_links(body_text)
        created_date, modified_date = extract_file_dates(filepath, fm_dict)

        lines = raw_text.splitlines()
        total_lines = len(lines)

        chunk_size = 80
        chunks: list[tuple[int, int, str]] = []

        if total_lines <= 100:
            chunks.append((1, max(1, total_lines), raw_text))
        else:
            for start_idx in range(0, total_lines, chunk_size):
                end_idx = min(start_idx + chunk_size, total_lines)
                chunk_text = "\n".join(lines[start_idx:end_idx])
                chunks.append((start_idx + 1, end_idx, chunk_text))

        safe_filename = re.sub(r"[^A-Za-z0-9_]", "_", filename)

        for start_line, end_line, chunk_text in chunks:
            item_id = f"note_{safe_filename}_L{start_line}_L{end_line}_{uuid.uuid4().hex[:6]}"

            searchable_content = (
                f"Note Title: {title}\n"
                f"File: {rel_path}\n"
                + (f"Tags: {', '.join(tags)}\n" if tags else "")
                + (f"Headings: {', '.join(headings)}\n" if headings else "")
                + (f"Links: {', '.join(links)}\n" if links else "")
                + f"\n{chunk_text}"
            )

            meta_data = {
                "filepath": rel_path,
                "filename": filename,
                "extension": ext,
                "start_line": start_line,
                "end_line": end_line,
                "title": title,
                "headings": headings,
                "tags": tags,
                "links": links,
                "created_date": created_date,
                "modified_date": modified_date,
                "date": modified_date,
                "class_names": [],
                "function_names": [],
                "file_class_names": [],
                "defined_symbols": tags + headings + [title],
                "imported_symbols": links,
            }

            items.append(
                EmbeddingItem(
                    id=item_id,
                    source="notes",
                    content=searchable_content,
                    meta_data=meta_data,
                )
            )

        return items

    def scan_notes(self) -> list[EmbeddingItem]:
        """Recursively scans notes_dir for .md and .txt files and returns EmbeddingItems."""
        items: list[EmbeddingItem] = []
        if not os.path.exists(self.notes_dir):
            logger.warning(f"Notes directory '{self.notes_dir}' does not exist.")
            return items

        logger.info(f"Scanning notes directory at '{self.notes_dir}'...")

        ignored_dirs = {
            ".git", "node_modules", ".next", "__pycache__", ".venv", "venv",
            ".checkpoints", "outputs", "logs", ".idea", ".vscode"
        }

        note_files: list[str] = []
        for root, dirs, files in os.walk(self.notes_dir):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
            for file in files:
                if file.startswith("."):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    note_files.append(os.path.join(root, file))

        for filepath in note_files:
            file_items = self.process_file(filepath, self.notes_dir)
            items.extend(file_items)

        logger.info(
            f"Notes scan complete for '{self.notes_dir}': "
            f"{len(note_files)} files -> {len(items)} EmbeddingItems."
        )
        return items
