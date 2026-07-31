import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import ExperimentHarness
from app.extraction.entity_extractor import EntityExtractor

async def run_experiment():
    harness = ExperimentHarness(experiment_name="entity_extraction_job")
    harness.logger.info("Starting Entity Extraction GPU Experiment...")

    extractor = EntityExtractor()
    sample_text = """
    EchoMind is built with FastAPI, PostgreSQL, Qdrant, and Neo4j.
    Created by Google and Microsoft developers on 2026-07-31. Version v1.0.0.
    """
    
    start_time = time.perf_counter()
    entities = await extractor.extract_entities(sample_text, source_document_id="exp_doc_001")
    elapsed = time.perf_counter() - start_time

    records = [e.model_dump() for e in entities]
    harness.save_output_jsonl("extracted_entities.jsonl", records)

    entity_types = {e.type for e in entities}
    harness.print_benchmark_summary(
        total_items=1,
        processed_items=len(entities),
        elapsed_seconds=elapsed,
        extra_metrics={
            "Extracted Entities": len(entities),
            "Unique Entity Types": len(entity_types)
        }
    )

if __name__ == "__main__":
    asyncio.run(run_experiment())
