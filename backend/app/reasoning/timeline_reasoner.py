from typing import Any
from app.schemas.timeline import TimelineEventResponse

class TimelineReasoner:
    """
    Reasoning service analyzing chronological progression of project events and milestones.
    """
    async def analyze_project_evolution(self, events: list[TimelineEventResponse]) -> dict[str, Any]:
        """Calculates project progression metrics, velocity, and key milestones."""
        if not events:
            return {"status": "no_events", "milestones_count": 0, "evolution_stages": []}

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        high_importance = [e for e in sorted_events if e.importance_score >= 0.75]

        return {
            "total_events": len(sorted_events),
            "milestones_count": len(high_importance),
            "first_recorded_event": sorted_events[0].timestamp.isoformat(),
            "latest_recorded_event": sorted_events[-1].timestamp.isoformat(),
            "key_milestones": [e.title for e in high_importance],
        }
