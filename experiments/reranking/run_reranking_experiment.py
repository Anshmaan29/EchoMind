import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import ExperimentHarness
from app.embeddings.factory import embedding_provider
from app.vector.factory import vector_store
from app.vector.base import VectorRecord

async def run_experiment():
    harness = ExperimentHarness(experiment_name="reranking_experiment_job")
    harness.logger.info("Starting Vector Re-ranking & MMR GPU Experiment...")

    # Embed sample queries
    sample_texts = [
        "EchoMind combines vector embeddings with knowledge graph structures.",
        "PostgreSQL 16 is used for metadata database persistence.",
        "Qdrant stores high-dimensional dense vector embeddings."
    ]

    embeddings = await embedding_provider.embed_texts(sample_texts)
    
    records = []
    for idx, (txt, vec) in enumerate(zip(sample_texts, embeddings)):
        records.append(
            VectorRecord(
                id=f"exp_rec_{idx}",
                vector=vec,
                payload={"content": txt, "index": idx}
            )
        )

    await vector_store.initialize_collection(
        collection_name="echomind_experiments_rerank",
        dimension=embedding_provider.dimension
    )
    await vector_store.upsert_records("echomind_experiments_rerank", records)

    query = "How does EchoMind store vector embeddings?"
    query_vec = await embedding_provider.embed_single(query)

    start_time = time.perf_counter()
    results = await vector_store.search(
        collection_name="echomind_experiments_rerank",
        query_vector=query_vec,
        limit=3
    )
    elapsed = time.perf_counter() - start_time

    output_records = [
        {
            "query": query,
            "id": r.id,
            "score": r.score,
            "content": r.payload.get("content", "")
        }
        for r in results
    ]

    harness.save_output_jsonl("reranking_results.jsonl", output_records)

    harness.print_benchmark_summary(
        total_items=len(sample_texts),
        processed_items=len(results),
        elapsed_seconds=elapsed,
        extra_metrics={
            "Target Query       ": query,
            "Top Similarity Score": results[0].score if results else 0.0
        }
    )

if __name__ == "__main__":
    asyncio.run(run_experiment())
