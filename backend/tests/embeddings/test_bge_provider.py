import pytest
import numpy as np
from app.embeddings.bge_provider import BGEEmbeddingProvider

@pytest.mark.asyncio
async def test_bge_provider_dimension_and_normalization() -> None:
    provider = BGEEmbeddingProvider(dimension=1024)
    assert provider.dimension == 1024

    sample_texts = [
        "EchoMind Digital Memory Operating System",
        "BAAI BGE-M3 Production Embedding Model Integration"
    ]

    embeddings = await provider.embed_texts(sample_texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    assert len(embeddings[1]) == 1024

    # Verify L2 normalization (vector length == 1.0)
    norm0 = np.linalg.norm(embeddings[0])
    norm1 = np.linalg.norm(embeddings[1])
    assert abs(norm0 - 1.0) < 1e-4
    assert abs(norm1 - 1.0) < 1e-4

@pytest.mark.asyncio
async def test_bge_provider_benchmarks() -> None:
    provider = BGEEmbeddingProvider(dimension=1024)
    _ = await provider.embed_texts(["Benchmarking BGE-M3 performance telemetry."])
    
    benchmarks = provider.get_benchmarks()
    assert "device" in benchmarks
    assert "average_batch_latency_ms" in benchmarks
    assert "embeddings_per_sec" in benchmarks
    assert "peak_gpu_memory_mb" in benchmarks
    assert benchmarks["total_embeddings_generated"] >= 1
