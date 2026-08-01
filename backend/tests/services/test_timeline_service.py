"""
Tests for Phase 2.2 — TimelineService, TimelineEvent, DailySummary, and
TimelineAwareAskService.

All tests use mocked SearchService and GitConnector so no real embeddings
or git repositories are required.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.search_service import SearchResult
from app.services.timeline_service import (
    PERIOD_MONTH,
    PERIOD_TODAY,
    PERIOD_WEEK,
    DailySummary,
    TimelineEvent,
    TimelineService,
    _classify_source,
    _extract_timestamp,
    detect_temporal_intent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_result(
    filepath: str = "backend/app/services/search_service.py",
    filename: str = "search_service.py",
    source: str = "code",
    content: str = "class SearchService: ...",
    start_line: int = 1,
    end_line: int = 10,
    meta_data: dict | None = None,
) -> SearchResult:
    return SearchResult(
        id="test_id",
        filepath=filepath,
        filename=filename,
        start_line=start_line,
        end_line=end_line,
        score=0.85,
        content=content,
        source=source,
        meta_data=meta_data or {},
    )


def _make_git_result(commit_date: str = "2026-08-01T10:00:00+00:00") -> SearchResult:
    return _make_search_result(
        filepath="git://commit/abc1234567890",
        filename="commit_abc12345",
        source="timeline",
        content="Git commit abc12345\nMessage: Add timeline service\nFiles: app/services/timeline_service.py",
        meta_data={
            "commit": "abc1234567890",
            "short_hash": "abc12345",
            "author": "Dev",
            "author_email": "dev@example.com",
            "date": commit_date,
            "date_human": "2026-08-01 10:00 UTC",
            "branch": "main",
            "message": "Add timeline service",
            "files_changed": ["app/services/timeline_service.py"],
            "added_lines": 42,
            "deleted_lines": 0,
        },
    )


# ---------------------------------------------------------------------------
# TimelineEvent — model tests
# ---------------------------------------------------------------------------

class TestTimelineEvent:
    def test_construction_with_all_fields(self) -> None:
        now = datetime.now(timezone.utc)
        ev = TimelineEvent(
            timestamp=now,
            source="git",
            title="Add timeline service",
            summary="Adds TimelineService class to services module",
            filepath="git://commit/abc123",
            metadata={"branch": "main"},
        )
        assert ev.source == "git"
        assert ev.title == "Add timeline service"
        assert ev.filepath == "git://commit/abc123"
        assert ev.metadata["branch"] == "main"

    def test_time_label_format(self) -> None:
        ts = datetime(2026, 8, 1, 10, 30, 0, tzinfo=timezone.utc)
        ev = TimelineEvent(
            timestamp=ts, source="git", title="T", summary="S", filepath="/f"
        )
        assert ev.time_label == "10:30 UTC"

    def test_date_label_format(self) -> None:
        ts = datetime(2026, 8, 1, 10, 30, 0, tzinfo=timezone.utc)
        ev = TimelineEvent(
            timestamp=ts, source="code", title="T", summary="S", filepath="/f"
        )
        assert ev.date_label == "2026-08-01"

    def test_source_emoji_git(self) -> None:
        ev = TimelineEvent(
            timestamp=datetime.now(timezone.utc),
            source="git", title="T", summary="S", filepath="/f",
        )
        assert ev.source_emoji == "🔖"

    def test_source_emoji_code(self) -> None:
        ev = TimelineEvent(
            timestamp=datetime.now(timezone.utc),
            source="code", title="T", summary="S", filepath="/f",
        )
        assert ev.source_emoji == "🐍"

    def test_source_emoji_markdown(self) -> None:
        ev = TimelineEvent(
            timestamp=datetime.now(timezone.utc),
            source="markdown", title="T", summary="S", filepath="/f",
        )
        assert ev.source_emoji == "📄"

    def test_from_search_result_git(self) -> None:
        res = _make_git_result()
        ev = TimelineEvent.from_search_result(res)
        assert ev.source == "git"
        assert "Add timeline service" in ev.title
        assert ev.filepath.startswith("git://")

    def test_from_search_result_code(self) -> None:
        res = _make_search_result(filepath="app/services/foo.py", source="code")
        ev = TimelineEvent.from_search_result(res)
        assert ev.source == "code"
        assert "foo.py" in ev.title

    def test_from_search_result_markdown(self) -> None:
        res = _make_search_result(filepath="README.md", source="code")
        ev = TimelineEvent.from_search_result(res)
        assert ev.source == "markdown"

    def test_from_search_result_doc(self) -> None:
        res = _make_search_result(filepath="docs/architecture.txt", source="code")
        ev = TimelineEvent.from_search_result(res)
        assert ev.source == "doc"


# ---------------------------------------------------------------------------
# classify_source
# ---------------------------------------------------------------------------

class TestClassifySource:
    def test_timeline_source_label(self) -> None:
        res = _make_search_result(source="timeline")
        assert _classify_source(res) == "git"

    def test_git_source_label(self) -> None:
        res = _make_search_result(source="git")
        assert _classify_source(res) == "git"

    def test_markdown_extension(self) -> None:
        res = _make_search_result(filepath="docs/README.md", source="code")
        assert _classify_source(res) == "markdown"

    def test_doc_extension(self) -> None:
        res = _make_search_result(filepath="docs/guide.txt", source="code")
        assert _classify_source(res) == "doc"

    def test_code_extension(self) -> None:
        res = _make_search_result(filepath="app/main.py", source="code")
        assert _classify_source(res) == "code"

    def test_unknown_extension_defaults_to_code(self) -> None:
        res = _make_search_result(filepath="app/file.xyz", source="code")
        assert _classify_source(res) == "code"


# ---------------------------------------------------------------------------
# extract_timestamp
# ---------------------------------------------------------------------------

class TestExtractTimestamp:
    def test_iso_date_with_timezone(self) -> None:
        res = _make_search_result(
            meta_data={"date": "2026-08-01T10:00:00+00:00"}
        )
        ts = _extract_timestamp(res)
        assert ts.year == 2026
        assert ts.month == 8
        assert ts.day == 1

    def test_fallback_to_now_when_no_date(self) -> None:
        before = datetime.now(timezone.utc)
        res = _make_search_result(meta_data={})
        ts = _extract_timestamp(res)
        after = datetime.now(timezone.utc)
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# detect_temporal_intent
# ---------------------------------------------------------------------------

class TestDetectTemporalIntent:
    @pytest.mark.parametrize("question,expected", [
        ("What did I work on today?", PERIOD_TODAY),
        ("what changed today", PERIOD_TODAY),
        ("What did I build yesterday?", PERIOD_TODAY),
        ("What did I work on this week?", PERIOD_WEEK),
        ("what changed last week?", PERIOD_WEEK),
        ("what happened this month?", PERIOD_MONTH),
        ("What changed last month?", PERIOD_MONTH),
        ("What is SearchService?", None),
        ("Explain the RAG pipeline", None),
        ("How does the embedding work?", None),
        ("Show me recently added code", PERIOD_TODAY),
        ("what was done recently", PERIOD_TODAY),
    ])
    def test_period_detection(self, question: str, expected: str | None) -> None:
        assert detect_temporal_intent(question) == expected


# ---------------------------------------------------------------------------
# DailySummary
# ---------------------------------------------------------------------------

class TestDailySummary:
    def _make_git_event(self, hour: int = 10) -> TimelineEvent:
        ts = datetime(2026, 8, 1, hour, 0, 0, tzinfo=timezone.utc)
        return TimelineEvent(
            timestamp=ts,
            source="git",
            title=f"Commit at hour {hour}",
            summary="Some commit",
            filepath="git://commit/abc",
            metadata={"files_changed": ["app/main.py", "README.md"]},
        )

    def test_to_text_includes_date(self) -> None:
        summary = DailySummary(date=date(2026, 8, 1))
        text = summary.to_text()
        assert "2026-08-01" in text

    def test_to_text_with_commits(self) -> None:
        ev = self._make_git_event()
        summary = DailySummary(
            date=date(2026, 8, 1),
            commits=[ev],
            files_changed=["app/main.py"],
        )
        text = summary.to_text()
        assert "Commits" in text
        assert "app/main.py" in text

    def test_to_text_with_modules_and_classes(self) -> None:
        summary = DailySummary(
            date=date(2026, 8, 1),
            major_modules=["app.services"],
            new_classes=["TimelineService"],
        )
        text = summary.to_text()
        assert "app.services" in text
        assert "TimelineService" in text

    def test_empty_summary(self) -> None:
        summary = DailySummary(date=date(2026, 8, 1))
        text = summary.to_text()
        assert "0" in text  # 0 commits, 0 files


# ---------------------------------------------------------------------------
# TimelineService — unit tests with mocked dependencies
# ---------------------------------------------------------------------------

class TestTimelineService:
    def _make_service(
        self,
        search_results: list[SearchResult] | None = None,
        git_commits: list | None = None,
    ) -> TimelineService:
        """Creates a TimelineService with mocked search and git."""
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=search_results or [])

        svc = TimelineService(search_service=mock_search, repo_path="/tmp/nonexistent")

        # Patch GitConnector to return controlled commits
        if git_commits is not None:
            svc._load_git_events = MagicMock(return_value=git_commits)

        return svc

    @pytest.mark.asyncio
    async def test_get_events_for_period_today_empty(self) -> None:
        svc = self._make_service(search_results=[], git_commits=[])
        events = await svc.get_events_for_period(period=PERIOD_TODAY)
        assert isinstance(events, list)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_get_events_for_period_invalid(self) -> None:
        svc = self._make_service()
        with pytest.raises(ValueError, match="Unknown period"):
            await svc.get_events_for_period(period="invalid_period")

    @pytest.mark.asyncio
    async def test_get_events_for_period_returns_sorted(self) -> None:
        now = datetime.now(timezone.utc)
        ev_old = TimelineEvent(
            timestamp=now - timedelta(hours=5),
            source="code", title="Old", summary="s", filepath="/old.py",
        )
        ev_new = TimelineEvent(
            timestamp=now - timedelta(hours=1),
            source="code", title="New", summary="s", filepath="/new.py",
        )
        svc = self._make_service(git_commits=[])
        # Mock _load_search_events to return events in reverse order
        svc._load_search_events = AsyncMock(return_value=[ev_new, ev_old])
        events = await svc.get_events_for_period(period=PERIOD_TODAY)
        assert events[0].title == "Old"
        assert events[1].title == "New"

    @pytest.mark.asyncio
    async def test_get_events_source_filter(self) -> None:
        now = datetime.now(timezone.utc)
        git_ev = TimelineEvent(
            timestamp=now - timedelta(hours=1),
            source="git", title="Commit", summary="s", filepath="git://c/abc",
        )
        code_ev = TimelineEvent(
            timestamp=now - timedelta(hours=2),
            source="code", title="Code", summary="s", filepath="/app.py",
        )
        svc = self._make_service(git_commits=[])
        svc._load_search_events = AsyncMock(return_value=[git_ev, code_ev])
        events = await svc.get_events_for_period(period=PERIOD_TODAY, sources=["git"])
        assert all(e.source == "git" for e in events)

    @pytest.mark.asyncio
    async def test_build_daily_summary_returns_dataclass(self) -> None:
        svc = self._make_service(search_results=[], git_commits=[])
        svc._load_search_events = AsyncMock(return_value=[])
        # Pass empty events to avoid a second pipeline run
        summary = await svc.build_daily_summary(target_date=date(2026, 8, 1), events=[])
        assert isinstance(summary, DailySummary)
        assert summary.date == date(2026, 8, 1)

    @pytest.mark.asyncio
    async def test_build_daily_summary_files_from_git(self) -> None:
        now = datetime.now(timezone.utc)
        git_ev = TimelineEvent(
            timestamp=now,
            source="git",
            title="Commit",
            summary="s",
            filepath="git://c/abc",
            metadata={"files_changed": ["app/services/timeline_service.py", "README.md"]},
        )
        svc = self._make_service(git_commits=[])
        # Pass pre-fetched events directly — no second pipeline run
        summary = await svc.build_daily_summary(target_date=now.date(), events=[git_ev])
        assert "app/services/timeline_service.py" in summary.files_changed
        assert "README.md" in summary.files_changed

    def test_format_events_as_timeline_empty(self) -> None:
        svc = self._make_service()
        output = svc.format_events_as_timeline([], period_label="TODAY")
        assert "No activity found" in output
        assert "TODAY" in output

    def test_format_events_as_timeline_with_events(self) -> None:
        now = datetime.now(timezone.utc)
        ev = TimelineEvent(
            timestamp=now,
            source="git",
            title="Add feature",
            summary="Adds the new feature",
            filepath="git://c/abc",
        )
        svc = self._make_service()
        output = svc.format_events_as_timeline([ev], period_label="TODAY")
        assert "Add feature" in output
        assert "git" in output
        assert "Total:" in output

    @pytest.mark.asyncio
    async def test_period_window_today(self) -> None:
        since, until = TimelineService._period_window(PERIOD_TODAY)
        diff = until - since
        assert 23 <= diff.total_seconds() / 3600 <= 25

    @pytest.mark.asyncio
    async def test_period_window_week(self) -> None:
        since, until = TimelineService._period_window(PERIOD_WEEK)
        diff = until - since
        assert 6 <= diff.days <= 7

    @pytest.mark.asyncio
    async def test_period_window_month(self) -> None:
        since, until = TimelineService._period_window(PERIOD_MONTH)
        diff = until - since
        assert 29 <= diff.days <= 30


# ---------------------------------------------------------------------------
# TimelineAwareAskService — temporal routing tests
# ---------------------------------------------------------------------------

class TestTimelineAwareAskService:
    """Tests that TimelineAwareAskService routes temporal questions correctly."""

    @pytest.mark.asyncio
    async def test_non_temporal_question_delegates_to_base(self) -> None:
        from app.services.ask_service import AskResponse, TimelineAwareAskService

        base_response = AskResponse(
            question="What is SearchService?",
            answer="SearchService is a hybrid search class.",
            evidence=[],
        )

        svc = TimelineAwareAskService.__new__(TimelineAwareAskService)
        svc.search_service = MagicMock()
        svc._timeline_service = None

        # Patch the parent ask()
        import app.services.ask_service as ask_module
        with patch.object(ask_module.AskService, "ask", new=AsyncMock(return_value=base_response)):
            response = await svc.ask("What is SearchService?")

        assert response.answer == "SearchService is a hybrid search class."

    @pytest.mark.asyncio
    async def test_temporal_question_uses_timeline_context(self) -> None:
        from app.services.ask_service import AskResponse, EvidenceItem, TimelineAwareAskService

        # Build a mock timeline service
        mock_timeline = MagicMock()
        mock_timeline.get_events_for_period = AsyncMock(return_value=[])
        mock_timeline.build_daily_summary = AsyncMock(
            return_value=DailySummary(date=date(2026, 8, 1))
        )
        mock_timeline.format_events_as_timeline = MagicMock(
            return_value="=== TIMELINE ===\n  No activity found\n==="
        )

        mock_llm = MagicMock()
        mock_llm.generate_answer = AsyncMock(
            return_value="Today you worked on the timeline service."
        )

        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=[])

        svc = TimelineAwareAskService(
            search_service=mock_search,
            llm_provider_inst=mock_llm,
            timeline_service=mock_timeline,
        )

        response = await svc.ask("What did I work on today?")
        assert isinstance(response, AskResponse)
        assert "timeline" in response.answer.lower() or "worked" in response.answer.lower()
        mock_timeline.get_events_for_period.assert_called_once_with(period=PERIOD_TODAY)

    @pytest.mark.asyncio
    async def test_temporal_question_degrades_gracefully_on_error(self) -> None:
        from app.services.ask_service import AskResponse, TimelineAwareAskService

        # Timeline service raises an exception
        mock_timeline = MagicMock()
        mock_timeline.get_events_for_period = AsyncMock(side_effect=RuntimeError("Timeline failure"))

        base_response = AskResponse(
            question="What did I work on today?",
            answer="Fallback answer",
            evidence=[],
        )

        svc = TimelineAwareAskService.__new__(TimelineAwareAskService)
        svc.search_service = MagicMock()
        svc._timeline_service = mock_timeline

        import app.services.ask_service as ask_module
        with patch.object(ask_module.AskService, "ask", new=AsyncMock(return_value=base_response)):
            response = await svc.ask("What did I work on today?")

        # Should gracefully fall back to base AskService
        assert response.answer == "Fallback answer"
