"""
EchoMind Git History Indexing CLI.

Usage:
    python -m app.cli.git_index
    python -m app.cli.git_index --repo ../..  --max-commits 200
    python -m app.cli.git_index --since 2026-01-01
"""
import argparse
import asyncio
import os

from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.embeddings.factory import EmbeddingFactory
from app.embeddings.pipeline import GenericEmbeddingPipeline
from app.ingestion.git_connector import GitConnector


async def main_async(args: argparse.Namespace) -> None:
    setup_logging()

    repo_path = os.path.abspath(args.repo)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        logger.warning(
            f"'{repo_path}' does not appear to be a git repository (.git dir not found)."
        )

    logger.info(
        f"Starting EchoMind Git Memory Indexer — repo: '{repo_path}', "
        f"max commits: {args.max_commits}"
        + (f", since: {args.since}" if args.since else "")
    )

    # Load git commits as EmbeddingItems
    connector = GitConnector(repo_path=repo_path)
    items = connector.to_embedding_items(
        max_commits=args.max_commits,
        since=args.since or None,
    )

    if not items:
        print("\nNo git commits found to index.")
        return

    # Run through the existing embedding pipeline (no new pipeline code)
    provider_name = args.provider or settings.EMBEDDING_PROVIDER
    provider = EmbeddingFactory.get_provider(provider_name=provider_name)

    pipeline = GenericEmbeddingPipeline(
        embedder=provider,
        backup_filepath=args.backup or "data/embeddings_backup.jsonl",
    )
    metrics = await pipeline.process_items(
        items=items,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        max_workers=args.workers,
        resume=not args.no_resume,
    )

    device_info = getattr(provider, "device", "cpu").upper()
    model_name  = getattr(provider, "model_name", "hash_mock")

    print("\n" + "=" * 65)
    print("🔖 ECHOMIND GIT MEMORY INDEX SUMMARY")
    print("=" * 65)
    print(f"Repository           : {repo_path}")
    print(f"Commits Scanned      : {args.max_commits} (max)")
    print(f"EmbeddingItems       : {metrics.total_items}")
    print(f"Processed            : {metrics.processed_items}")
    print(f"Skipped (checkpoint) : {metrics.skipped_items}")
    print(f"Failed               : {metrics.failed_items}")
    print(f"Embedding Provider   : {provider_name.upper()}")
    print(f"Embedding Model      : {model_name}")
    print(f"Device               : {device_info}")
    print(f"Embeddings/sec       : {metrics.embeddings_per_sec}")
    print("=" * 65)
    print("\n✅ Git history indexed. Run 'python -m app.cli.ask --query \"What changed today?\"'")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoMind Git Memory Indexer — index Git commit history into the embedding pipeline"
    )
    parser.add_argument(
        "--repo", "-r",
        default="..",
        help="Path to the local Git repository root (default: .. i.e. repo root when run from backend/)",
    )
    parser.add_argument(
        "--max-commits", "-n",
        type=int,
        default=500,
        help="Maximum number of commits to index, most recent first (default: 500)",
    )
    parser.add_argument(
        "--since", "-s",
        default=None,
        help="Only index commits on or after this date (ISO format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--provider", "-p",
        default=None,
        help="Embedding provider override (mock, qwen, openai, bge)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of async worker tasks (default: 4)",
    )
    parser.add_argument(
        "--collection-name", "-c",
        default=settings.QDRANT_COLLECTION_NAME,
        help="Target Qdrant collection name",
    )
    parser.add_argument(
        "--backup",
        default=None,
        help="JSONL backup filepath (default: data/embeddings_backup.jsonl)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resumption and reprocess all commits",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
