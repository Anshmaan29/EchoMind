"""
EchoMind Personal Notes Memory Indexer CLI — Phase 3.0

Indexes Markdown (.md) and text (.txt) notes into the embedding pipeline.

Usage:
    python -m app.cli.notes_index
    python -m app.cli.notes_index --notes-dir ../notes
"""
import argparse
import asyncio
import os

from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.embeddings.factory import EmbeddingFactory
from app.embeddings.pipeline import GenericEmbeddingPipeline
from app.ingestion.note_connector import NoteConnector


async def main_async(args: argparse.Namespace) -> None:
    setup_logging()

    connector = NoteConnector(notes_dir=args.notes_dir)
    notes_dir = connector.notes_dir
    logger.info(f"Starting EchoMind Personal Notes Indexer — directory: '{notes_dir}'")

    items = connector.scan_notes()

    if not items:
        print(f"\nNo notes (.md, .txt) found to index in '{notes_dir}'.")
        return

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
    model_name = getattr(provider, "model_name", "hash_mock")

    print("\n" + "=" * 65)
    print("📝 ECHOMIND PERSONAL NOTES INDEX SUMMARY")
    print("=" * 65)
    print(f"Notes Directory     : {notes_dir}")
    print(f"EmbeddingItems      : {metrics.total_items}")
    print(f"Processed           : {metrics.processed_items}")
    print(f"Skipped (checkpoint): {metrics.skipped_items}")
    print(f"Failed              : {metrics.failed_items}")
    print(f"Embedding Provider  : {provider_name.upper()}")
    print(f"Embedding Model     : {model_name}")
    print(f"Device              : {device_info}")
    print("=" * 65)
    print("\n✅ Notes indexed. Ask questions like: 'python -m app.cli.ask --query \"What notes do I have about RAG?\"'")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoMind Personal Notes Indexer — index Markdown & text notes into the embedding pipeline"
    )
    parser.add_argument(
        "--notes-dir", "-n",
        default=None,
        help="Path to local notes directory (default: auto-discovered <project_root>/notes)",
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
        help="Disable checkpoint resumption and reprocess all notes",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
