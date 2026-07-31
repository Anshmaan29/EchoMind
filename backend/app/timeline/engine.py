import re
from datetime import datetime, timezone
from app.schemas.entity import EntityCreate
from app.schemas.timeline import TimelineEventCreate
from app.timeline.base import BaseTimelineEngine

class TimelineEngine(BaseTimelineEngine):
    """
    Temporal Timeline Engine extracting chronological events and project evolution milestones.
    """
    async def create_timeline_events(
        self,
        text: str,
        entities: list[EntityCreate],
        source_document_id: str | None = None
    ) -> list[TimelineEventCreate]:
        events: list[TimelineEventCreate] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        entity_names = [e.name for e in entities]
        project_names = [e.name for e in entities if e.type == "Project"] or ["EchoMind System"]

        for idx, sent in enumerate(sentences):
            sent_clean = sent.strip()
            if not sent_clean or len(sent_clean) < 15:
                continue

            # Look for dates or milestone keywords (created, launched, updated, released, milestone, feature)
            milestone_keywords = [
                "created", "launched", "released", "updated", "deployed",
                "milestone", "feature", "architecture", "initial", "version", "started"
            ]
            has_milestone = any(kw in sent_clean.lower() for kw in milestone_keywords)

            if has_milestone or idx == 0:
                event_title = sent_clean[:80] + ("..." if len(sent_clean) > 80 else "")
                importance = 0.90 if has_milestone else 0.50

                matched_entities = [name for name in entity_names if name.lower() in sent_clean.lower()]

                events.append(
                    TimelineEventCreate(
                        title=event_title,
                        description=sent_clean,
                        timestamp=datetime.now(timezone.utc),
                        entities_involved=matched_entities or entity_names[:5],
                        projects_involved=project_names,
                        importance_score=importance,
                        source_document_id=source_document_id,
                        meta_data={"sentence_index": idx}
                    )
                )

        return events
