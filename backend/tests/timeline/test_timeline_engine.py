import pytest
from app.extraction.entity_extractor import EntityExtractor
from app.timeline.engine import TimelineEngine

@pytest.mark.asyncio
async def test_timeline_event_creation() -> None:
    entity_extractor = EntityExtractor()
    timeline_engine = TimelineEngine()

    sample_text = "EchoMind Milestone 1 was launched successfully. We created the initial architecture and deployed PostgreSQL."
    entities = await entity_extractor.extract_entities(sample_text)

    events = await timeline_engine.create_timeline_events(
        text=sample_text,
        entities=entities,
        source_document_id="doc_123"
    )

    assert len(events) >= 1
    assert events[0].importance_score >= 0.50
    assert events[0].source_document_id == "doc_123"
