"""
EchoMind Timeline Intelligence Service — Phase 2.2

Aggregates project activity from the existing embedding/search layer into a
chronological timeline.  No new external connectors — reuses SearchService
and GitConnector exclusively.

Sources merged:
  • git      — commits indexed via GitConnector / git_index CLI
  • code     — Python/JS/TS chunks embedded via embed CLI
  • markdown — .md files (README, docs, notes)
  • doc      — other documentation files (.txt, .rst, .pdf text)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import logger
from app.ingestion.git_connector import GitConnector
from app.services.search_service import SearchResult, SearchService


# ---------------------------------------------------------------------------
# Period constants
# ---------------------------------------------------------------------------

PERIOD_TODAY = "today"
PERIOD_WEEK  = "week"
PERIOD_MONTH = "month"

_PERIOD_DAYS: dict[str, int] = {
    PERIOD_TODAY: 1,
    PERIOD_WEEK:  7,
    PERIOD_MONTH: 30,
}

# ---------------------------------------------------------------------------
# Source label helpers
# ---------------------------------------------------------------------------

_GIT_SOURCE_LABELS   = {"timeline", "git"}
_CODE_EXTENSIONS     = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".cpp", ".c", ".h"}
_MARKDOWN_EXTENSIONS = {".md", ".mdx"}
_DOC_EXTENSIONS      = {".txt", ".rst", ".pdf", ".html", ".htm"}


def _classify_source(result: SearchResult) -> str:
    """
    Maps a SearchResult to a human-readable source label.

    Priority:
      1. result.source field  (e.g. 'timeline' from GitConnector)
      2. file extension of filepath
      3. fallback to 'code'
    """
    if result.source.lower() in _GIT_SOURCE_LABELS:
        return "git"

    ext = os.path.splitext(result.filepath)[1].lower()
    if ext in _MARKDOWN_EXTENSIONS:
        return "markdown"
    if ext in _DOC_EXTENSIONS:
        return "doc"
    return "code"


def _extract_timestamp(result: SearchResult) -> datetime:
    """
    Extracts the best available timestamp from a SearchResult.

    For git chunks: parses the 'date' field in metadata.
    For code/doc chunks: falls back to the current UTC time.
    """
    meta = result.meta_data or {}
    date_str = meta.get("date") or meta.get("date_human", "")
    if date_str:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M UTC",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

    return datetime.now(timezone.utc)


def _build_title(result: SearchResult, source_label: str) -> str:
    """Short human-readable title for a timeline event."""
    meta = result.meta_data or {}
    if source_label == "git":
        msg = meta.get("message", "").strip()
        if msg:
            return msg[:90] + ("..." if len(msg) > 90 else "")
        return f"Commit {meta.get('short_hash', result.filename)}"
    name = result.filename or os.path.basename(result.filepath)
    return f"{name} (L{result.start_line}-L{result.end_line})"


def _build_summary(result: SearchResult, source_label: str) -> str:
    """One-to-two sentence summary for display and LLM context injection."""
    meta = result.meta_data or {}
    if source_label == "git":
        msg     = meta.get("message", "").strip()
        files   = meta.get("files_changed", [])
        added   = meta.get("added_lines", 0)
        deleted = meta.get("deleted_lines", 0)
        parts: list[str] = []
        if msg:
            parts.append(msg)
        if files:
            sample = ", ".join(files[:3])
            suffix = f" (+{len(files) - 3} more)" if len(files) > 3 else ""
            parts.append(f"Changed: {sample}{suffix}")
        if added or deleted:
            parts.append(f"+{added}/-{deleted} lines")
        return "  ".join(parts) if parts else result.content[:200]
    text = result.content.strip()
    return text[:200] + ("..." if len(text) > 200 else "")


def _build_summary_from_git_commit(commit: Any) -> str:
    """Helper: builds a summary string from a GitCommit object."""
    parts: list[str] = []
    msg = getattr(commit, "message", "").strip()
    if msg:
        parts.append(msg)
    files = getattr(commit, "files_changed", [])
    if files:
        sample = ", ".join(files[:3])
        suffix = f" (+{len(files) - 3} more)" if len(files) > 3 else ""
        parts.append(f"Changed: {sample}{suffix}")
    added   = getattr(commit, "added_lines", 0)
    deleted = getattr(commit, "deleted_lines", 0)
    if added or deleted:
        parts.append(f"+{added}/-{deleted} lines")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# TimelineEvent — the canonical model
# ---------------------------------------------------------------------------

class TimelineEvent(BaseModel):
    """
    Chronological project activity event built from the embedding search layer.

    Fields
    ------
    timestamp : UTC datetime of the event
    source    : 'git' | 'code' | 'markdown' | 'doc'
    title     : Human-readable one-liner
    summary   : 1-2 sentence description
    filepath  : Originating file path
    metadata  : Source-specific extras (commit hash, branch, changed files, etc.)
    """
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime             = Field(description="UTC datetime of the event")
    source:    str                  = Field(description="git | code | markdown | doc")
    title:     str                  = Field(description="Human-readable one-liner")
    summary:   str                  = Field(description="1-2 sentence description")
    filepath:  str                  = Field(description="Originating file path")
    metadata:  dict[str, Any]       = Field(default_factory=dict, description="Source-specific extras")

    @property
    def time_label(self) -> str:
        """Formatted HH:MM UTC string."""
        return self.timestamp.strftime("%H:%M UTC")

    @property
    def date_label(self) -> str:
        """Formatted YYYY-MM-DD string."""
        return self.timestamp.strftime("%Y-%m-%d")

    @property
    def source_emoji(self) -> str:
        return {
            "git":      "🔖",
            "code":     "🐍",
            "markdown": "📄",
            "doc":      "📚",
        }.get(self.source, "🗂️")

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "TimelineEvent":
        """Factory: converts a SearchResult into a TimelineEvent."""
        source_label = _classify_source(result)
        return cls(
            timestamp=_extract_timestamp(result),
            source=source_label,
            title=_build_title(result, source_label),
            summary=_build_summary(result, source_label),
            filepath=result.filepath,
            metadata=result.meta_data or {},
        )


# ---------------------------------------------------------------------------
# DailySummary
# ---------------------------------------------------------------------------

@dataclass
class DailySummary:
    """
    Aggregated summary of all project activity for a single calendar day.
    """
    date:                  date
    commits:               list[TimelineEvent] = field(default_factory=list)
    files_changed:         list[str]           = field(default_factory=list)
    major_modules:         list[str]           = field(default_factory=list)
    new_classes:           list[str]           = field(default_factory=list)
    deleted_files:         list[str]           = field(default_factory=list)
    documentation_updates: list[str]           = field(default_factory=list)

    def to_text(self) -> str:
        """Formats the summary as a plain-text block for LLM injection and CLI display."""
        lines: list[str] = []
        lines.append(f"TODAY'S WORK SUMMARY  --  {self.date}")
        lines.append("-" * 52)
        lines.append(f"  Commits          : {len(self.commits)}")
        lines.append(f"  Files changed    : {len(self.files_changed)}")

        if self.commits:
            lines.append("\n  Git commits:")
            for ev in self.commits[:10]:
                lines.append(f"    * [{ev.time_label}]  {ev.title}")

        if self.files_changed:
            lines.append("\n  Changed files:")
            for fp in self.files_changed[:15]:
                lines.append(f"    * {fp}")
            if len(self.files_changed) > 15:
                lines.append(f"    ... and {len(self.files_changed) - 15} more")

        if self.major_modules:
            lines.append("\n  Major modules touched:")
            for m in self.major_modules[:10]:
                lines.append(f"    * {m}")

        if self.new_classes:
            lines.append("\n  New classes / functions:")
            for c in self.new_classes[:10]:
                lines.append(f"    * {c}")

        if self.deleted_files:
            lines.append("\n  Deleted files:")
            for df in self.deleted_files[:10]:
                lines.append(f"    * {df}")

        if self.documentation_updates:
            lines.append("\n  Documentation updates:")
            for doc in self.documentation_updates[:10]:
                lines.append(f"    * {doc}")

        lines.append("-" * 52)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Temporal query helpers
# ---------------------------------------------------------------------------

_TEMPORAL_QUERIES: dict[str, list[str]] = {
    PERIOD_TODAY: [
        "commits today work done today",
        "code changes today files modified today",
        "documentation updated today",
    ],
    PERIOD_WEEK: [
        "commits this week work done this week",
        "code changes this week files modified this week",
        "documentation updated this week",
    ],
    PERIOD_MONTH: [
        "commits this month work done this month",
        "code changes this month major features this month",
    ],
}

# Regex: detect temporal intent in natural-language questions
_TEMPORAL_INTENT_RE = re.compile(
    r"\b(today|yesterday|this\s+week|last\s+week|this\s+month|last\s+month"
    r"|what\s+did\s+i\s+(work\s+on|build|do|create|change|write)"
    r"|what\s+(changed|happened|was\s+done)"
    r"|recent(ly)?|latest)\b",
    re.IGNORECASE,
)
_WEEK_RE  = re.compile(r"\b(this|last)\s+week\b", re.IGNORECASE)
_MONTH_RE = re.compile(r"\b(this|last)\s+month\b", re.IGNORECASE)


def detect_temporal_intent(question: str) -> str | None:
    """
    Returns a period string (PERIOD_TODAY / PERIOD_WEEK / PERIOD_MONTH) if the
    question contains temporal intent, or None for general questions.
    """
    if not _TEMPORAL_INTENT_RE.search(question):
        return None
    if _MONTH_RE.search(question):
        return PERIOD_MONTH
    if _WEEK_RE.search(question):
        return PERIOD_WEEK
    return PERIOD_TODAY


# ---------------------------------------------------------------------------
# TimelineService
# ---------------------------------------------------------------------------

class TimelineService:
    """
    EchoMind Timeline Intelligence Service.

    Aggregates project activity from the existing SearchService embedding
    store into chronological TimelineEvent lists.  Works entirely from
    the search layer — no database required.

    Usage::

        service = TimelineService()
        events  = await service.get_events_for_period("today")
        summary = await service.build_daily_summary(date.today())
    """

    def __init__(
        self,
        search_service: SearchService | None = None,
        repo_path: str | None = None,
        top_k_per_query: int = 20,
    ) -> None:
        self.search_service  = search_service or SearchService()
        self.repo_path       = repo_path or self._discover_repo_path()
        self.top_k_per_query = top_k_per_query

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _discover_repo_path() -> str:
        candidates = [
            "../..",
            "..",
            "/Users/anshmaansingh/Echomind",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        ]
        for p in candidates:
            if os.path.isdir(os.path.join(p, ".git")):
                return os.path.abspath(p)
        return ".."

    @staticmethod
    def _period_window(period: str) -> tuple[datetime, datetime]:
        """Returns (since_dt, until_dt) for a named period."""
        now  = datetime.now(timezone.utc)
        days = _PERIOD_DAYS.get(period, 1)
        return now - timedelta(days=days), now

    @staticmethod
    def _event_in_window(event: TimelineEvent, since: datetime, until: datetime) -> bool:
        ts = event.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return since <= ts <= until

    # ------------------------------------------------------------------ #
    # Git-direct path (accurate timestamps)
    # ------------------------------------------------------------------ #

    def _load_git_events(self, since: datetime) -> list[TimelineEvent]:
        """
        Loads git commits directly via GitConnector (subprocess) for accurate
        timestamps.  Converts each GitCommit into a TimelineEvent.
        """
        try:
            connector = GitConnector(repo_path=self.repo_path)
            since_str = since.strftime("%Y-%m-%d")
            commits   = connector.load_commits(max_commits=200, since=since_str)
            events: list[TimelineEvent] = []

            for commit in commits:
                # Parse timestamp — always normalise to UTC so window filtering works
                dt: datetime | None = None
                for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d"):
                    try:
                        parsed = datetime.strptime(commit.date_iso, fmt)
                        # Convert any tz-aware datetime to UTC; make naive datetimes UTC
                        dt = parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                if dt is None:
                    dt = datetime.now(timezone.utc)

                msg = commit.message.strip()
                events.append(TimelineEvent(
                    timestamp=dt,
                    source="git",
                    title=msg[:90] + ("..." if len(msg) > 90 else ""),
                    summary=_build_summary_from_git_commit(commit),
                    filepath=f"git://commit/{commit.commit_hash}",
                    metadata={
                        "commit":        commit.commit_hash,
                        "short_hash":    commit.short_hash,
                        "author":        commit.author_name,
                        "author_email":  commit.author_email,
                        "date":          commit.date_iso,
                        "date_human":    commit.date_human,
                        "branch":        commit.branch,
                        "message":       commit.message,
                        "files_changed": commit.files_changed,
                        "added_lines":   commit.added_lines,
                        "deleted_lines": commit.deleted_lines,
                    },
                ))
            return events
        except Exception as exc:
            logger.warning(f"TimelineService: git events load failed: {exc}")
            return []

    # ------------------------------------------------------------------ #
    # Search-based path (code / markdown / doc events)
    # ------------------------------------------------------------------ #

    async def _load_search_events(
        self,
        period: str,
        since: datetime,
        until: datetime,
    ) -> list[TimelineEvent]:
        """
        Uses SearchService to pull code/markdown/doc events via temporal queries.
        Git events from the search index are excluded here (we use the direct path).
        """
        queries = _TEMPORAL_QUERIES.get(period, _TEMPORAL_QUERIES[PERIOD_TODAY])
        seen_keys: set[tuple[str, int, int]] = set()
        events: list[TimelineEvent] = []

        for q in queries:
            try:
                results = await self.search_service.search(
                    query=q,
                    top_k=self.top_k_per_query,
                    min_score=0.0,
                )
                for res in results:
                    key = (res.filepath, res.start_line, res.end_line)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    ev = TimelineEvent.from_search_result(res)
                    # Skip git events here — already fetched via GitConnector
                    if ev.source != "git":
                        events.append(ev)
            except Exception as exc:
                logger.warning(f"TimelineService: search query '{q}' failed: {exc}")

        return events

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def get_events(
        self,
        since: datetime,
        until: datetime,
        sources: list[str] | None = None,
        period: str = PERIOD_TODAY,
    ) -> list[TimelineEvent]:
        """
        Returns chronologically sorted TimelineEvent list for a time window.

        :param since:   Start of window (UTC-aware datetime).
        :param until:   End of window (UTC-aware datetime).
        :param sources: Optional filter list: ['git', 'code', 'markdown', 'doc']
        :param period:  Named period string used to select temporal queries.
        :return: Sorted list of TimelineEvent objects (oldest first).
        """
        git_events    = self._load_git_events(since=since)
        search_events = await self._load_search_events(period=period, since=since, until=until)
        all_events    = git_events + search_events

        # Apply window filter for git events (GitConnector may return older commits)
        all_events = [ev for ev in all_events if self._event_in_window(ev, since, until)]

        # Source filter
        if sources:
            all_events = [ev for ev in all_events if ev.source in sources]

        # Sort chronologically (oldest first)
        all_events.sort(key=lambda ev: ev.timestamp)

        logger.info(
            f"TimelineService: {len(all_events)} events "
            f"[git={sum(1 for e in all_events if e.source == 'git')}, "
            f"code={sum(1 for e in all_events if e.source == 'code')}, "
            f"markdown={sum(1 for e in all_events if e.source == 'markdown')}, "
            f"doc={sum(1 for e in all_events if e.source == 'doc')}]"
        )
        return all_events

    async def get_events_for_period(
        self,
        period: str = PERIOD_TODAY,
        sources: list[str] | None = None,
    ) -> list[TimelineEvent]:
        """
        Convenience wrapper: returns events for a named period.

        :param period: 'today' | 'week' | 'month'
        :param sources: Optional source filter.
        :return: Sorted list of TimelineEvent objects.
        """
        if period not in _PERIOD_DAYS:
            raise ValueError(f"Unknown period '{period}'. Use: {list(_PERIOD_DAYS)}")
        since, until = self._period_window(period)
        return await self.get_events(since=since, until=until, sources=sources, period=period)

    async def build_daily_summary(
        self,
        target_date: date | None = None,
        events: list["TimelineEvent"] | None = None,
    ) -> DailySummary:
        """
        Generates a DailySummary for a given calendar day.

        Pass ``events`` (from a prior ``get_events_for_period`` call) to avoid
        a second full git+search pipeline execution.  When omitted the method
        fetches events itself using a rolling 24-hour window so that git
        commits are found regardless of calendar-day UTC boundary issues.

        :param target_date: Calendar date (defaults to today UTC).
        :param events:      Pre-fetched events to aggregate (avoids double execution).
        :return: DailySummary object.
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc).date()

        if events is None:
            # Use a rolling 24-hour window (same as PERIOD_TODAY) so that git
            # --since boundaries don't differ between this call and a prior
            # get_events_for_period() call.
            since, until = self._period_window(PERIOD_TODAY)
            events = await self.get_events(since=since, until=until, period=PERIOD_TODAY)

        git_events  = [ev for ev in events if ev.source == "git"]
        code_events = [ev for ev in events if ev.source == "code"]
        doc_events  = [ev for ev in events if ev.source in ("markdown", "doc")]

        # Changed files from git commits
        files_changed: list[str] = []
        for ev in git_events:
            for fp in ev.metadata.get("files_changed", []):
                if fp not in files_changed:
                    files_changed.append(fp)

        # Module names from code events
        major_modules: list[str] = []
        for ev in code_events:
            mod   = os.path.splitext(ev.filepath)[0].replace(os.sep, ".").lstrip(".")
            parts = mod.split(".")
            label = ".".join(parts[-2:]) if len(parts) >= 2 else mod
            if label and label not in major_modules:
                major_modules.append(label)

        # Class / function names from code event metadata
        new_classes: list[str] = []
        for ev in code_events:
            for sym_key in ("class_names", "function_names", "defined_symbols"):
                for sym in ev.metadata.get(sym_key, []):
                    if sym and sym not in new_classes:
                        new_classes.append(sym)

        doc_updates = list({ev.filepath for ev in doc_events})

        return DailySummary(
            date=target_date,
            commits=git_events,
            files_changed=files_changed,
            major_modules=major_modules[:20],
            new_classes=new_classes[:30],
            deleted_files=[],
            documentation_updates=doc_updates[:20],
        )

    def format_events_as_timeline(
        self,
        events: list[TimelineEvent],
        period_label: str = "TODAY",
    ) -> str:
        """
        Formats a list of TimelineEvent objects as a rich CLI timeline string.

        :param events:       Sorted list of timeline events.
        :param period_label: Display label for the header (TODAY / THIS WEEK / etc.).
        :return: Formatted multi-line string.
        """
        now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lines: list[str] = []

        lines.append("=" * 58)
        lines.append(f"==> {period_label} TIMELINE  --  {now_label}")
        lines.append("=" * 58)

        if not events:
            lines.append("")
            lines.append("  No activity found for this period.")
            lines.append(
                "  Tip: run 'python -m app.cli.git_index' to index git history first."
            )
            lines.append("=" * 58)
            return "\n".join(lines)

        for ev in events:
            lines.append("")
            lines.append(f"{ev.source_emoji}  [{ev.time_label}]  {ev.source} -- {ev.title}")
            lines.append(f"    {ev.filepath}")
            brief = ev.summary.replace("\n", " ")[:120]
            if brief:
                lines.append(f"    {brief}")

        lines.append("")
        lines.append("-" * 58)
        git_n  = sum(1 for e in events if e.source == "git")
        code_n = sum(1 for e in events if e.source == "code")
        md_n   = sum(1 for e in events if e.source in ("markdown", "doc"))
        lines.append(
            f"  Total: {len(events)} events  "
            f"[git={git_n}  code={code_n}  docs={md_n}]"
        )
        lines.append("=" * 58)
        return "\n".join(lines)
