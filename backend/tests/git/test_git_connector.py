"""
Tests for GitConnector — git history reading, metadata extraction,
EmbeddingItem conversion, and CLI integration.

All tests use a temporary in-process git repo so they run offline
with zero network access and no external dependencies.
"""
import os
import subprocess
import tempfile
import pytest

from app.ingestion.git_connector import GitCommit, GitConnector
from app.embeddings.pipeline import EmbeddingItem


# ---------------------------------------------------------------------------
# Fixture: fresh git repo with a few commits
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_git_repo(tmp_path):
    """Creates a real minimal git repository for testing."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=str(repo), check=True,
                       capture_output=True, text=True)

    git("init", "-b", "main")
    git("config", "user.email", "test@echomind.ai")
    git("config", "user.name", "EchoMind Test")

    # Commit 1
    (repo / "README.md").write_text("# EchoMind\nInitial commit.\n")
    git("add", "README.md")
    git("commit", "-m", "docs: initial README")

    # Commit 2
    (repo / "app.py").write_text("class EmbeddingFactory:\n    pass\n")
    (repo / "provider.py").write_text("class QwenEmbeddingProvider:\n    pass\n")
    git("add", "app.py", "provider.py")
    git("commit", "-m", "feat: add EmbeddingFactory and QwenEmbeddingProvider")

    # Commit 3
    (repo / "search.py").write_text("class SearchService:\n    pass\n")
    git("add", "search.py")
    git("commit", "-m", "feat: add SearchService")

    return str(repo)


# ---------------------------------------------------------------------------
# Unit tests — GitConnector
# ---------------------------------------------------------------------------

def test_load_commits_count(tmp_git_repo):
    connector = GitConnector(repo_path=tmp_git_repo)
    commits = connector.load_commits(max_commits=100)
    assert len(commits) == 3, f"Expected 3 commits, got {len(commits)}"


def test_commit_fields_populated(tmp_git_repo):
    connector = GitConnector(repo_path=tmp_git_repo)
    commits = connector.load_commits(max_commits=1)
    c = commits[0]  # Most recent
    assert c.commit_hash and len(c.commit_hash) == 40
    assert c.author_name == "EchoMind Test"
    assert c.author_email == "test@echomind.ai"
    assert c.date_iso  # non-empty
    assert c.message   # non-empty
    assert isinstance(c.files_changed, list)


def test_commit_most_recent_first(tmp_git_repo):
    connector = GitConnector(repo_path=tmp_git_repo)
    commits = connector.load_commits()
    messages = [c.message for c in commits]
    assert messages[0] == "feat: add SearchService"
    assert messages[-1] == "docs: initial README"


def test_files_changed_populated(tmp_git_repo):
    connector = GitConnector(repo_path=tmp_git_repo)
    commits = connector.load_commits()
    # Second-most-recent commit touched app.py and provider.py
    commit_with_two_files = [c for c in commits if "EmbeddingFactory" in c.message][0]
    assert len(commit_with_two_files.files_changed) == 2
    assert any("app.py" in f for f in commit_with_two_files.files_changed)
    assert any("provider.py" in f for f in commit_with_two_files.files_changed)


def test_numstat_added_lines(tmp_git_repo):
    connector = GitConnector(repo_path=tmp_git_repo)
    commits = connector.load_commits()
    # All commits added lines
    for c in commits:
        assert c.added_lines >= 0
        assert c.deleted_lines >= 0


def test_max_commits_limit(tmp_git_repo):
    connector = GitConnector(repo_path=tmp_git_repo)
    commits = connector.load_commits(max_commits=2)
    assert len(commits) == 2


# ---------------------------------------------------------------------------
# Unit tests — GitCommit model
# ---------------------------------------------------------------------------

def _make_commit(**kwargs) -> GitCommit:
    defaults = dict(
        commit_hash="a" * 40,
        author_name="Alice",
        author_email="alice@example.com",
        date_iso="2026-08-01T00:00:00+00:00",
        branch="main",
        message="feat: hybrid search",
        files_changed=["backend/app/services/search_service.py"],
        added_lines=120,
        deleted_lines=15,
    )
    defaults.update(kwargs)
    return GitCommit(**defaults)


def test_short_hash():
    c = _make_commit()
    assert c.short_hash == "a" * 8


def test_date_human_format():
    c = _make_commit(date_iso="2026-08-01T05:30:00+05:30")
    assert "2026-08-01" in c.date_human
    assert "UTC" in c.date_human


def test_embedding_text_contains_key_fields():
    c = _make_commit(message="feat: hybrid retrieval engine")
    text = c.to_embedding_text()
    assert "feat: hybrid retrieval engine" in text
    assert "Alice" in text
    assert "2026-08-01" in text
    assert "search_service.py" in text
    assert "120" in text  # added lines
    assert "15" in text   # deleted lines


def test_to_embedding_item_structure():
    c = _make_commit()
    item: EmbeddingItem = c.to_embedding_item()
    assert item.source == "timeline"
    assert "commit/" in item.meta_data["filepath"]
    assert item.meta_data["commit"] == "a" * 40
    assert item.meta_data["author"] == "Alice"
    assert isinstance(item.meta_data["files_changed"], list)


# ---------------------------------------------------------------------------
# Integration test — to_embedding_items pipeline
# ---------------------------------------------------------------------------

def test_to_embedding_items_returns_list(tmp_git_repo):
    connector = GitConnector(repo_path=tmp_git_repo)
    items = connector.to_embedding_items(max_commits=100)
    assert len(items) == 3
    for item in items:
        assert isinstance(item, EmbeddingItem)
        assert item.source == "timeline"
        assert item.content  # non-empty
        assert "commit" in item.meta_data


def test_embedding_items_have_unique_ids(tmp_git_repo):
    connector = GitConnector(repo_path=tmp_git_repo)
    items = connector.to_embedding_items()
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids)), "Duplicate EmbeddingItem IDs found"
