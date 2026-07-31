import os
import pytest
from experiments.common import ExperimentHarness

def test_experiment_harness_initialization() -> None:
    harness = ExperimentHarness(experiment_name="test_harness")
    assert harness.device in ["cuda", "mps", "cpu"]
    assert os.path.exists(harness.output_dir)
    assert os.path.exists(harness.log_dir)

@pytest.mark.asyncio
async def test_run_embedding_experiment() -> None:
    from experiments.embedding_generation.run_embedding_experiment import run_experiment
    await run_experiment()
    assert os.path.exists("experiments/outputs/embedding_outputs.jsonl")

@pytest.mark.asyncio
async def test_run_entity_experiment() -> None:
    from experiments.entity_extraction.run_entity_experiment import run_experiment
    await run_experiment()
    assert os.path.exists("experiments/outputs/extracted_entities.jsonl")

@pytest.mark.asyncio
async def test_run_relationship_experiment() -> None:
    from experiments.relationship_extraction.run_relationship_experiment import run_experiment
    await run_experiment()
    assert os.path.exists("experiments/outputs/extracted_relationships.jsonl")

@pytest.mark.asyncio
async def test_run_reranking_experiment() -> None:
    from experiments.reranking.run_reranking_experiment import run_experiment
    await run_experiment()
    assert os.path.exists("experiments/outputs/reranking_results.jsonl")

@pytest.mark.asyncio
async def test_run_evaluation_experiment() -> None:
    from experiments.evaluation.run_evaluation_experiment import run_experiment
    await run_experiment()
    assert os.path.exists("experiments/outputs/evaluation_metrics.jsonl")
