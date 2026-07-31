import os
import pytest
from app.ingestion.github_connector import GitHubConnector

def test_github_connector_local_scan(tmp_path) -> None:
    connector = GitHubConnector()
    
    # Create mock repo folder with .git marker and source files
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    git_dir = repo_dir / ".git"
    git_dir.mkdir()

    py_file = repo_dir / "main.py"
    py_file.write_text("def hello():\n    print('EchoMind GitHub Connector Test')\n", encoding="utf-8")

    md_file = repo_dir / "README.md"
    md_file.write_text("# Mock Repository\n\nGitHub connector unit test.", encoding="utf-8")

    items = connector.scan_repository(str(repo_dir))
    assert len(items) >= 2

    for item in items:
        assert item.source == "github"
        assert "filepath" in item.meta_data
        assert "start_line" in item.meta_data
        assert "end_line" in item.meta_data
        assert "commit_hash" in item.meta_data
        assert "branch" in item.meta_data

@pytest.mark.asyncio
async def test_github_experiment_runner() -> None:
    from experiments.github_embedding.run_github_experiment import run_experiment
    await run_experiment()
    assert os.path.exists("experiments/outputs/embeddings_backup.jsonl")
