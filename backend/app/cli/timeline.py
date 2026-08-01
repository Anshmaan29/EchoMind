"""
EchoMind Timeline CLI — Phase 2.2

Prints a chronological activity timeline for today, this week, or this month.

Usage::

    python -m app.cli.timeline --today
    python -m app.cli.timeline --week
    python -m app.cli.timeline --month
    python -m app.cli.timeline --today --sources git code
    python -m app.cli.timeline --today --summary
"""
import argparse
import asyncio
from datetime import datetime, timezone

from app.core.logging import setup_logging
from app.services.timeline_service import (
    PERIOD_MONTH,
    PERIOD_TODAY,
    PERIOD_WEEK,
    TimelineService,
)


async def main_async(args: argparse.Namespace) -> None:
    setup_logging()

    # Determine period
    if args.month:
        period = PERIOD_MONTH
        period_label = "THIS MONTH"
    elif args.week:
        period = PERIOD_WEEK
        period_label = "THIS WEEK"
    else:
        period = PERIOD_TODAY
        period_label = "TODAY"

    # Optional source filter
    sources = args.sources if args.sources else None

    timeline_svc = TimelineService(top_k_per_query=args.top_k)

    # Fetch events — one call, reused for both display and summary
    events = await timeline_svc.get_events_for_period(period=period, sources=sources)

    # Print formatted timeline
    print()
    print(timeline_svc.format_events_as_timeline(events, period_label=period_label))

    # Daily summary — reuse already-fetched events (no second git/search run)
    if args.summary or period == PERIOD_TODAY:
        print()
        try:
            summary = await timeline_svc.build_daily_summary(events=events)
            print(summary.to_text())
        except Exception as exc:
            print(f"  [summary unavailable: {exc}]")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoMind Timeline CLI — chronological project activity viewer"
    )

    period_group = parser.add_mutually_exclusive_group()
    period_group.add_argument(
        "--today",
        action="store_true",
        default=False,
        help="Show timeline for the past 24 hours (default)",
    )
    period_group.add_argument(
        "--week",
        action="store_true",
        default=False,
        help="Show timeline for the past 7 days",
    )
    period_group.add_argument(
        "--month",
        action="store_true",
        default=False,
        help="Show timeline for the past 30 days",
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["git", "code", "markdown", "doc"],
        default=None,
        help="Filter by source type (default: all sources)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Append a daily summary block (always shown for --today)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Max search results per query (default: 20)",
    )

    args = parser.parse_args()

    # Default to --today if no period flag was provided
    if not args.week and not args.month:
        args.today = True

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
