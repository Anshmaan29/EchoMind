import pytest
import numpy as np
from app.embeddings.qwen_provider import QwenEmbeddingProvider

@pytest.mark.asyncio
async def test_qwen_provider_dimension_and_normalization() -> None:
    provider = QwenEmbeddingProvider(dimension=4096)
    assert provider.dimension == 4096

    sample_texts = [
        "EchoMind Digital Memory Operating System",
        "Qwen3-Embedding-8B Production Model Integration"
    ]

    embeddings = await provider.embed_texts(sample_texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 4096
    assert len(embeddings[1]) == 4096

    # Verify L2 normalization (vector length == 1.0)
    norm0 = np.linalg.norm(embeddings[0])
    norm1 = np.linalg.norm(embeddings[1])
    assert abs(norm0 - 1.0) < 1e-4
    assert abs(norm1 - 1.0) < 1e-4

@pytest.mark.asyncio
async def test_qwen_provider_benchmarks() -> None:
    provider = QwenEmbeddingProvider(dimension=4096)
    _ = await provider.embed_texts(["Benchmarking Qwen3-Embedding-8B performance telemetry."])
    
    benchmarks = provider.get_benchmarks()
    assert "device" in benchmarks
    assert "average_batch_latency_ms" in benchmarks
    assert "embeddings_per_sec" in benchmarks
    assert "peak_gpu_memory_mb" in benchmarks
    assert benchmarks["total_embeddings_generated"] >= 1
