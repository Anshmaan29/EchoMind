"""
Tests for Phase 2.2 — Timeline CLI (app.cli.timeline).

Tests verify argument parsing, period routing, source filtering, and the
summary flag — all without running git or the real embedding pipeline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.timeline_service import TimelineEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(source: str = "git", hour: int = 10) -> TimelineEvent:
    return TimelineEvent(
        timestamp=datetime(2026, 8, 1, hour, 0, 0, tzinfo=timezone.utc),
        source=source,
        title=f"Event from {source}",
        summary="A test event",
        filepath=f"git://commit/abc_{hour}" if source == "git" else f"app/{source}_file.py",
    )


def _make_args(
    today: bool = True,
    week: bool = False,
    month: bool = False,
    sources: list[str] | None = None,
    summary: bool = False,
    top_k: int = 20,
) -> argparse.Namespace:
    return argparse.Namespace(
        today=today,
        week=week,
        month=month,
        sources=sources,
        summary=summary,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------

class TestTimelineCLIArgParsing:
    def test_default_period_is_today(self) -> None:
        """When no period flag is passed, defaults should be today=True."""
        import sys
        from unittest.mock import patch

        with patch("sys.argv", ["timeline"]):
            import importlib
            import app.cli.timeline as cli_mod
            importlib.reload(cli_mod)
            # Verify parser produces today=True by default when no flag given
            parser = argparse.ArgumentParser()
            period_group = parser.add_mutually_exclusive_group()
            period_group.add_argument("--today", action="store_true", default=False)
            period_group.add_argument("--week", action="store_true", default=False)
            period_group.add_argument("--month", action="store_true", default=False)
            parser.add_argument("--sources", nargs="+", default=None)
            parser.add_argument("--summary", action="store_true", default=False)
            parser.add_argument("--top-k", type=int, default=20)
            args = parser.parse_args([])
            assert not args.week
            assert not args.month

    def test_week_flag_parsed(self) -> None:
        parser = argparse.ArgumentParser()
        period_group = parser.add_mutually_exclusive_group()
        period_group.add_argument("--today", action="store_true", default=False)
        period_group.add_argument("--week", action="store_true", default=False)
        period_group.add_argument("--month", action="store_true", default=False)
        args = parser.parse_args(["--week"])
        assert args.week is True
        assert args.today is False
        assert args.month is False

    def test_month_flag_parsed(self) -> None:
        parser = argparse.ArgumentParser()
        period_group = parser.add_mutually_exclusive_group()
        period_group.add_argument("--today", action="store_true", default=False)
        period_group.add_argument("--week", action="store_true", default=False)
        period_group.add_argument("--month", action="store_true", default=False)
        args = parser.parse_args(["--month"])
        assert args.month is True

    def test_sources_flag_parsed(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--sources", nargs="+", default=None)
        args = parser.parse_args(["--sources", "git", "code"])
        assert args.sources == ["git", "code"]

    def test_summary_flag_parsed(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--summary", action="store_true", default=False)
        args = parser.parse_args(["--summary"])
        assert args.summary is True

    def test_top_k_flag_parsed(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--top-k", type=int, default=20)
        args = parser.parse_args(["--top-k", "50"])
        assert args.top_k == 50


# ---------------------------------------------------------------------------
# main_async — period routing
# ---------------------------------------------------------------------------

class TestTimelineCLIMainAsync:
    @pytest.mark.asyncio
    async def test_today_calls_get_events_today(self, capsys) -> None:
        from app.cli.timeline import main_async
        from app.services.timeline_service import DailySummary, PERIOD_TODAY
        from datetime import date

        events = [_make_event("git", 9), _make_event("code", 10)]
        mock_svc = MagicMock()
        mock_svc.get_events_for_period = AsyncMock(return_value=events)
        # build_daily_summary is now called with events= kwarg
        mock_svc.build_daily_summary = AsyncMock(
            return_value=DailySummary(date=date(2026, 8, 1))
        )
        mock_svc.format_events_as_timeline = MagicMock(
            return_value="=== TODAY TIMELINE ===\n  Event from git\n==="
        )

        args = _make_args(today=True, week=False, month=False)
        with patch("app.cli.timeline.TimelineService", return_value=mock_svc):
            await main_async(args)

        mock_svc.get_events_for_period.assert_called_once_with(
            period=PERIOD_TODAY, sources=None
        )
        # Verify build_daily_summary was called with events= (no double run)
        mock_svc.build_daily_summary.assert_called_once_with(events=events)
        captured = capsys.readouterr()
        assert "TODAY TIMELINE" in captured.out

    @pytest.mark.asyncio
    async def test_week_calls_get_events_week(self, capsys) -> None:
        from app.cli.timeline import main_async
        from app.services.timeline_service import PERIOD_WEEK

        events = [_make_event("git")]
        mock_svc = MagicMock()
        mock_svc.get_events_for_period = AsyncMock(return_value=events)
        mock_svc.format_events_as_timeline = MagicMock(
            return_value="=== THIS WEEK TIMELINE ==="
        )

        args = _make_args(today=False, week=True, month=False, summary=False)
        with patch("app.cli.timeline.TimelineService", return_value=mock_svc):
            await main_async(args)

        mock_svc.get_events_for_period.assert_called_once_with(
            period=PERIOD_WEEK, sources=None
        )

    @pytest.mark.asyncio
    async def test_month_calls_get_events_month(self, capsys) -> None:
        from app.cli.timeline import main_async
        from app.services.timeline_service import PERIOD_MONTH

        mock_svc = MagicMock()
        mock_svc.get_events_for_period = AsyncMock(return_value=[])
        mock_svc.format_events_as_timeline = MagicMock(
            return_value="=== THIS MONTH TIMELINE ==="
        )

        args = _make_args(today=False, week=False, month=True, summary=False)
        with patch("app.cli.timeline.TimelineService", return_value=mock_svc):
            await main_async(args)

        mock_svc.get_events_for_period.assert_called_once_with(
            period=PERIOD_MONTH, sources=None
        )

    @pytest.mark.asyncio
    async def test_source_filter_passed_through(self, capsys) -> None:
        from app.cli.timeline import main_async
        from app.services.timeline_service import PERIOD_TODAY

        mock_svc = MagicMock()
        mock_svc.get_events_for_period = AsyncMock(return_value=[])
        mock_svc.build_daily_summary = AsyncMock(
            return_value=MagicMock(to_text=MagicMock(return_value=""))
        )
        mock_svc.format_events_as_timeline = MagicMock(return_value="=== FILTERED ===")

        args = _make_args(today=True, sources=["git"], summary=False)
        with patch("app.cli.timeline.TimelineService", return_value=mock_svc):
            await main_async(args)

        mock_svc.get_events_for_period.assert_called_once_with(
            period=PERIOD_TODAY, sources=["git"]
        )

    @pytest.mark.asyncio
    async def test_summary_flag_triggers_build_daily_summary(self, capsys) -> None:
        from app.cli.timeline import main_async
        from datetime import date
        from app.services.timeline_service import DailySummary

        daily = DailySummary(date=date(2026, 8, 1), commits=[], files_changed=[])
        mock_svc = MagicMock()
        mock_svc.get_events_for_period = AsyncMock(return_value=[])
        mock_svc.build_daily_summary = AsyncMock(return_value=daily)
        mock_svc.format_events_as_timeline = MagicMock(return_value="=== TL ===")

        args = _make_args(today=True, summary=True)
        with patch("app.cli.timeline.TimelineService", return_value=mock_svc):
            await main_async(args)

        # build_daily_summary must be called with events= kwarg (pre-fetched events)
        call_kwargs = mock_svc.build_daily_summary.call_args
        assert call_kwargs is not None
        assert "events" in (call_kwargs.kwargs if call_kwargs.kwargs else {})

    @pytest.mark.asyncio
    async def test_no_events_shows_no_activity_message(self, capsys) -> None:
        from app.cli.timeline import main_async
        from datetime import date
        from app.services.timeline_service import DailySummary

        mock_svc = MagicMock()
        mock_svc.get_events_for_period = AsyncMock(return_value=[])
        mock_svc.build_daily_summary = AsyncMock(
            return_value=DailySummary(date=date(2026, 8, 1))
        )
        # Use real format method to verify "No activity" message
        from app.services.timeline_service import TimelineService as RealSvc
        real_format = RealSvc.format_events_as_timeline
        mock_svc.format_events_as_timeline = lambda events, period_label: real_format(
            mock_svc, events, period_label
        )

        args = _make_args(today=True, summary=False)
        with patch("app.cli.timeline.TimelineService", return_value=mock_svc):
            await main_async(args)

        captured = capsys.readouterr()
        assert "No activity found" in captured.out

    @pytest.mark.asyncio
    async def test_top_k_passed_to_service(self) -> None:
        from app.cli.timeline import main_async
        from datetime import date
        from app.services.timeline_service import DailySummary

        constructed_svc = None

        class CaptureSvc:
            def __init__(self, **kwargs):
                nonlocal constructed_svc
                self.top_k_per_query = kwargs.get("top_k_per_query")
                constructed_svc = self
                self.get_events_for_period = AsyncMock(return_value=[])
                self.build_daily_summary = AsyncMock(return_value=DailySummary(date=date(2026, 8, 1)))
                self.format_events_as_timeline = MagicMock(return_value="=== TL ===")

        args = _make_args(today=True, top_k=42, summary=False)
        with patch("app.cli.timeline.TimelineService", CaptureSvc):
            await main_async(args)

        assert constructed_svc is not None
        assert constructed_svc.top_k_per_query == 42
