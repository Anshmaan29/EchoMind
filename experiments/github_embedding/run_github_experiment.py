import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import ExperimentHarness
from app.embeddings.factory import embedding_provider
from app.embeddings.pipeline import GenericEmbeddingPipeline
from app.ingestion.github_connector import GitHubConnector

async def run_experiment():
    harness = ExperimentHarness(experiment_name="github_embedding_job")
    harness.logger.info("Starting GitHub Repository Embedding GPU Experiment...")

    connector = GitHubConnector()
    # Scan local workspace repository
    repo_path = harness.root_dir
    items = connector.scan_repository(repo_path)

    if not items:
        harness.logger.warning(f"No source items discovered in repository '{repo_path}'.")
        return

    output_backup_path = os.path.join(harness.output_dir, "embeddings_backup.jsonl")
    checkpoint_path = os.path.join(harness.checkpoint_dir, "github_embedding_checkpoint.db")

    pipeline = GenericEmbeddingPipeline(
        embedder=embedding_provider,
        checkpoint_db_path=checkpoint_path,
        backup_filepath=output_backup_path
    )

    batch_size = harness.config.get("experiment", {}).get("batch_size", 64)
    max_workers = harness.config.get("experiment", {}).get("max_workers", 4)

    start_time = time.perf_counter()
    metrics = await pipeline.process_items(
        items=items,
        collection_name="echomind_github_repo_memories",
        batch_size=batch_size,
        max_workers=max_workers,
        resume=True
    )
    elapsed = time.perf_counter() - start_time

    git_meta = connector.extract_git_metadata(repo_path)

    harness.print_benchmark_summary(
        total_items=metrics.total_items,
        processed_items=metrics.processed_items,
        elapsed_seconds=elapsed,
        extra_metrics={
            "Repository Path    ": repo_path,
            "Git Commit Hash    ": git_meta.get("commit_hash", "unknown")[:10],
            "Git Branch         ": git_meta.get("branch", "main"),
            "Skipped (Check)    ": metrics.skipped_items,
            "Embeddings / Sec   ": metrics.embeddings_per_sec,
            "Backup JSONL Output": output_backup_path
        }
    )

if __name__ == "__main__":
    asyncio.run(run_experiment())
