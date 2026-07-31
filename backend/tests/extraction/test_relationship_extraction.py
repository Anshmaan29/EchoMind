import pytest
from app.extraction.entity_extractor import EntityExtractor
from app.extraction.relation_extractor import RelationshipExtractor

@pytest.mark.asyncio
async def test_relationship_extraction() -> None:
    entity_extractor = EntityExtractor()
    rel_extractor = RelationshipExtractor()

    sample_text = "EchoMind uses FastAPI and depends on PostgreSQL for database storage."
    entities = await entity_extractor.extract_entities(sample_text, source_document_id="doc_123")
    
    relationships = await rel_extractor.extract_relationships(
        text=sample_text,
        entities=entities,
        source_document_id="doc_123"
    )

    assert len(relationships) >= 1
    rel_types = {r.relation_type for r in relationships}
    assert ("USES" in rel_types or "DEPENDS_ON" in rel_types or "CONNECTED_TO" in rel_types)
