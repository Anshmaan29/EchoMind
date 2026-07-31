import argparse
import json
import pytest
from app.cli.search import main_async

@pytest.mark.asyncio
async def test_search_cli_execution(tmp_path) -> None:
    backup_file = tmp_path / "embeddings_backup.jsonl"
    rec = {
        "id": "cli_rec_01",
        "source": "github",
        "content": "def test_cli(): pass",
        "embedding_vector": [0.1] * 384,
        "metadata": {
            "filepath": "backend/app/cli/search.py",
            "filename": "search.py",
            "start_line": 5,
            "end_line": 20
        }
    }
    backup_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    args = argparse.Namespace(
        query="test cli search",
        top_k=3,
        backup_path=str(backup_file),
        min_score=0.0
    )

    await main_async(args)
