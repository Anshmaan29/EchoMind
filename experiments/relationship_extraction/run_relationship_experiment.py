import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import ExperimentHarness
from app.extraction.entity_extractor import EntityExtractor
from app.extraction.relation_extractor import RelationshipExtractor

async def run_experiment():
    harness = ExperimentHarness(experiment_name="relationship_extraction_job")
    harness.logger.info("Starting Relationship Extraction GPU Experiment...")

    entity_extractor = EntityExtractor()
    rel_extractor = RelationshipExtractor()

    sample_text = "EchoMind uses FastAPI and depends on PostgreSQL for metadata database storage."
    
    start_time = time.perf_counter()
    entities = await entity_extractor.extract_entities(sample_text, source_document_id="exp_doc_002")
    relationships = await rel_extractor.extract_relationships(
        text=sample_text,
        entities=entities,
        source_document_id="exp_doc_002"
    )
    elapsed = time.perf_counter() - start_time

    records = [r.model_dump() for r in relationships]
    harness.save_output_jsonl("extracted_relationships.jsonl", records)

    harness.print_benchmark_summary(
        total_items=1,
        processed_items=len(relationships),
        elapsed_seconds=elapsed,
        extra_metrics={
            "Input Entities    ": len(entities),
            "Discovered Edges  ": len(relationships)
        }
    )

if __name__ == "__main__":
    asyncio.run(run_experiment())
