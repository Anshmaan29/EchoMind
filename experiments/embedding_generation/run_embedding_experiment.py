import asyncio
import os
import sys
import time

# Ensure backend and experiments modules are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import ExperimentHarness
from app.embeddings.pipeline import EmbeddingItem, GenericEmbeddingPipeline
from app.embeddings.factory import embedding_provider

async def run_experiment():
    harness = ExperimentHarness(experiment_name="embedding_generation_job")
    harness.logger.info("Starting Embedding Generation GPU Experiment...")

    # Load dataset sample items
    data_dir = os.path.join(harness.root_dir, "data")
    sample_text = "EchoMind AI Kosh GPU embedding generation test passage."
    if os.path.exists(os.path.join(data_dir, "sample_knowledge.md")):
        with open(os.path.join(data_dir, "sample_knowledge.md"), "r", encoding="utf-8") as f:
            sample_text = f.read()

    paragraphs = [p.strip() for p in sample_text.split("\n\n") if p.strip()]
    items = [
        EmbeddingItem(
            id=f"exp_emb_{idx}",
            source="pdf",
            content=para,
            meta_data={"para_idx": idx}
        )
        for idx, para in enumerate(paragraphs)
    ]

    pipeline = GenericEmbeddingPipeline(
        embedder=embedding_provider,
        checkpoint_db_path=os.path.join(harness.checkpoint_dir, "embedding_checkpoint.db"),
        backup_filepath=os.path.join(harness.output_dir, "embedding_outputs.jsonl")
    )

    batch_size = harness.config.get("experiment", {}).get("batch_size", 64)
    max_workers = harness.config.get("experiment", {}).get("max_workers", 4)
    resume = harness.config.get("experiment", {}).get("resume", True)

    start_time = time.perf_counter()
    metrics = await pipeline.process_items(
        items=items,
        batch_size=batch_size,
        max_workers=max_workers,
        resume=resume
    )
    elapsed = time.perf_counter() - start_time

    harness.print_benchmark_summary(
        total_items=metrics.total_items,
        processed_items=metrics.processed_items,
        elapsed_seconds=elapsed,
        extra_metrics={
            "Skipped (Checkpoints)": metrics.skipped_items,
            "Failed Items        ": metrics.failed_items,
            "Embeddings / Sec    ": metrics.embeddings_per_sec
        }
    )

if __name__ == "__main__":
    asyncio.run(run_experiment())
