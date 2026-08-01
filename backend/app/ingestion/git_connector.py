"""
Git Memory Connector — reads local Git repository history via subprocess.

Extracts per-commit metadata (hash, author, date, branch, message, changed files,
added/deleted line counts) and converts each commit into an EmbeddingItem that
feeds into the existing embedding pipeline.  No new dependencies required.
"""
from __future__ import annotations

import re
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.logging import logger
from app.embeddings.pipeline import EmbeddingItem

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

class GitCommit:
    """Plain-data container for a single Git commit."""

    def __init__(
        self,
        commit_hash: str,
        author_name: str,
        author_email: str,
        date_iso: str,
        branch: str,
        message: str,
        files_changed: list[str],
        added_lines: int,
        deleted_lines: int,
    ) -> None:
        self.commit_hash = commit_hash
        self.author_name = author_name
        self.author_email = author_email
        self.date_iso = date_iso
        self.branch = branch
        self.message = message.strip()
        self.files_changed = files_changed
        self.added_lines = added_lines
        self.deleted_lines = deleted_lines

    # Convenience
    @property
    def short_hash(self) -> str:
        return self.commit_hash[:8]

    @property
    def date_human(self) -> str:
        """ISO 8601 -> 'YYYY-MM-DD HH:MM UTC'."""
        try:
            dt = datetime.fromisoformat(self.date_iso)
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return self.date_iso

    def to_embedding_text(self) -> str:
        """
        Produces a rich, searchable text representation of the commit.
        Temporal keywords (today, yesterday, date) are embedded naturally so
        that queries like 'What changed today?' resolve via hybrid search.
        """
        files_str = "\n".join(f"  - {f}" for f in self.files_changed) or "  (no files)"
        return (
            f"Git Commit {self.short_hash}\n"
            f"Author: {self.author_name} <{self.author_email}>\n"
            f"Date: {self.date_human}\n"
            f"Branch: {self.branch}\n"
            f"Message: {self.message}\n"
            f"Files changed ({len(self.files_changed)}):\n{files_str}\n"
            f"Lines added: {self.added_lines} | Lines deleted: {self.deleted_lines}"
        )

    def to_embedding_item(self) -> EmbeddingItem:
        """Converts this commit into an EmbeddingItem for the pipeline."""
        item_id = f"git_{self.short_hash}_{uuid.uuid4().hex[:6]}"
        return EmbeddingItem(
            id=item_id,
            source="timeline",
            content=self.to_embedding_text(),
            meta_data={
                "filepath": f"git://commit/{self.commit_hash}",
                "filename": f"commit_{self.short_hash}",
                "extension": ".git",
                "start_line": 1,
                "end_line": 1,
                # Git-specific metadata
                "commit": self.commit_hash,
                "short_hash": self.short_hash,
                "author": self.author_name,
                "author_email": self.author_email,
                "date": self.date_iso,
                "date_human": self.date_human,
                "branch": self.branch,
                "message": self.message,
                "files_changed": self.files_changed,
                "added_lines": self.added_lines,
                "deleted_lines": self.deleted_lines,
                # Hybrid-search helpers (treated same as code symbols)
                "class_names": [],
                "function_names": [],
                "file_class_names": [],
                "defined_symbols": [self.short_hash, self.message[:40]],
                "imported_symbols": [],
            },
        )


# ---------------------------------------------------------------------------
# Git Connector
# ---------------------------------------------------------------------------

class GitConnector:
    """
    Reads local Git repository history using subprocess + git CLI.
    Compatible with all Git versions; no external Python dependencies.
    """

    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = repo_path

    def _run(self, cmd: list[str], check: bool = True) -> str:
        """Run a git command and return stdout as a stripped string."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check,
                encoding="utf-8",
                errors="replace",
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as exc:
            logger.warning(f"git command failed: {' '.join(cmd)}: {exc.stderr.strip()}")
            return ""

    def current_branch(self) -> str:
        """Returns the current branch name (falls back to 'HEAD' for detached state)."""
        branch = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"], check=False)
        return branch or "HEAD"

    def _changed_files(self, commit_hash: str) -> list[str]:
        """Returns list of file paths changed in a commit."""
        out = self._run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash],
            check=False,
        )
        return [f.strip() for f in out.splitlines() if f.strip()]

    def _numstat(self, commit_hash: str) -> tuple[int, int]:
        """Returns (added_lines, deleted_lines) for a commit."""
        out = self._run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--numstat", commit_hash],
            check=False,
        )
        added = deleted = 0
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    added += int(parts[0]) if parts[0] != "-" else 0
                    deleted += int(parts[1]) if parts[1] != "-" else 0
                except ValueError:
                    pass
        return added, deleted

    def load_commits(
        self,
        max_commits: int = 500,
        since: str | None = None,
    ) -> list[GitCommit]:
        """
        Reads Git history and returns a list of GitCommit objects.

        :param max_commits: Maximum number of commits to load (most-recent-first).
        :param since: Optional ISO date string. Only commits on or after this date.
        :return: List of GitCommit objects.
        """
        log_format = "%x00".join(["%H", "%an", "%ae", "%aI", "%D", "%s"])
        cmd = ["git", "log", f"--format={log_format}", f"-n{max_commits}"]
        if since:
            cmd.append(f"--since={since}")

        raw = self._run(cmd, check=False)
        if not raw:
            logger.warning(f"No git log output from repo '{self.repo_path}'.")
            return []

        commits: list[GitCommit] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\x00")
            if len(parts) < 6:
                continue

            commit_hash, author_name, author_email, date_iso, refs, message = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            )

            # Resolve branch from refs field (e.g. "HEAD -> main, origin/main")
            branch = "unknown"
            for ref in refs.split(","):
                ref = ref.strip()
                m = re.match(r"(?:HEAD -> )?(.+)", ref)
                if m and "HEAD" not in m.group(1):
                    branch = m.group(1).strip()
                    break
            if branch == "unknown":
                branch = self.current_branch()

            files_changed = self._changed_files(commit_hash)
            added_lines, deleted_lines = self._numstat(commit_hash)

            commits.append(
                GitCommit(
                    commit_hash=commit_hash,
                    author_name=author_name,
                    author_email=author_email,
                    date_iso=date_iso,
                    branch=branch,
                    message=message,
                    files_changed=files_changed,
                    added_lines=added_lines,
                    deleted_lines=deleted_lines,
                )
            )

        logger.info(f"GitConnector: loaded {len(commits)} commits from '{self.repo_path}'.")
        return commits

    def to_embedding_items(
        self,
        max_commits: int = 500,
        since: str | None = None,
    ) -> list[EmbeddingItem]:
        """
        Convenience wrapper that loads commits and converts them to EmbeddingItems.

        :param max_commits: Maximum number of commits to load.
        :param since: Optional ISO date string filter.
        :return: List of EmbeddingItem objects ready for the pipeline.
        """
        commits = self.load_commits(max_commits=max_commits, since=since)
        items = [c.to_embedding_item() for c in commits]
        logger.info(f"GitConnector: produced {len(items)} EmbeddingItems from git history.")
        return items
