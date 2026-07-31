from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.timeline import TimelineEvent
from app.schemas.timeline import TimelineEventResponse, TimelineResponse

class TimelineService:
    """Application Service for managing and retrieving project evolution timeline events."""
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def get_timeline(self, limit: int = 50) -> TimelineResponse:
        stmt = select(TimelineEvent).order_by(TimelineEvent.timestamp.desc()).limit(limit)
        res = await self.db_session.execute(stmt)
        events = res.scalars().all()

        event_responses = [TimelineEventResponse.model_validate(e) for e in events]
        return TimelineResponse(
            project_name=None,
            events=event_responses,
            total_events=len(event_responses)
        )

    async def get_project_timeline(self, project_name: str, limit: int = 50) -> TimelineResponse:
        stmt = select(TimelineEvent).order_by(TimelineEvent.timestamp.asc()).limit(limit)
        res = await self.db_session.execute(stmt)
        events = res.scalars().all()

        # Filter events involving the project
        project_lower = project_name.lower()
        filtered = [
            e for e in events
            if any(project_lower in str(p).lower() for p in (e.projects_involved or [])) or project_lower in e.title.lower() or project_lower in e.description.lower()
        ]

        event_responses = [TimelineEventResponse.model_validate(e) for e in (filtered or events)]
        return TimelineResponse(
            project_name=project_name,
            events=event_responses,
            total_events=len(event_responses)
        )
