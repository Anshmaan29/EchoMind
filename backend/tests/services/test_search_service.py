import json
import pytest
from app.embeddings.mock_provider import MockEmbeddingProvider
from app.services.search_service import SearchService

@pytest.mark.asyncio
async def test_search_service_jsonl_backup(tmp_path) -> None:
    backup_file = tmp_path / "embeddings_backup.jsonl"
    
    embedder = MockEmbeddingProvider(dimension=384)
    vec1 = await embedder.embed_single("FastAPI backend architecture and endpoints")
    vec2 = await embedder.embed_single("Database migrations with Alembic and SQLAlchemy")

    rec1 = {
        "id": "rec_001",
        "source": "github",
        "content": "class SearchService: async def search(...)",
        "embedding_vector": vec1,
        "metadata": {
            "filepath": "backend/app/services/search_service.py",
            "filename": "search_service.py",
            "start_line": 1,
            "end_line": 30
        }
    }

    rec2 = {
        "id": "rec_002",
        "source": "github",
        "content": "async def get_db_session(): yield session",
        "embedding_vector": vec2,
        "metadata": {
            "filepath": "backend/app/database/session.py",
            "filename": "session.py",
            "start_line": 10,
            "end_line": 25
        }
    }

    backup_file.write_text(f"{json.dumps(rec1)}\n{json.dumps(rec2)}\n", encoding="utf-8")

    search_service = SearchService(embedder=embedder, backup_filepath=str(backup_file))
    results = await search_service.search(query="FastAPI backend architecture", top_k=2)

    assert len(results) >= 1
    assert results[0].id == "rec_001"
    assert results[0].filepath == "backend/app/services/search_service.py"
    assert results[0].start_line == 1
    assert results[0].end_line == 30
    assert results[0].score >= 0.0
