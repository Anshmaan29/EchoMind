import pytest
from app.extraction.entity_extractor import EntityExtractor

@pytest.mark.asyncio
async def test_entity_extraction_types() -> None:
    extractor = EntityExtractor()
    sample_text = """
    In EchoMind, we created a FastAPI backend using Python, PostgreSQL, and Qdrant.
    Version v1.0.0 was deployed by Google and Microsoft on 2026-07-31.
    Check issue PR-42 or visit https://github.com/echomind for details.
    """

    entities = await extractor.extract_entities(sample_text, source_document_id="doc_123")
    assert len(entities) > 0

    entity_types = {e.type for e in entities}
    assert "Programming Language" in entity_types  # Python
    assert "Framework" in entity_types             # FastAPI
    assert "Technology" in entity_types            # PostgreSQL / Qdrant
    assert "Version" in entity_types               # v1.0.0
    assert "URL" in entity_types                   # https://github.com/echomind
