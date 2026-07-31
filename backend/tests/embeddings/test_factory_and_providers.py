import pytest
import numpy as np
from app.core.exceptions import EchoMindException
from app.embeddings.factory import EmbeddingFactory, get_embedding_provider
from app.embeddings.mock_provider import MockEmbeddingProvider

def test_factory_default_mock_provider() -> None:
    provider = EmbeddingFactory.get_provider(provider_name="mock", dimension=384)
    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.dimension == 384

def test_get_embedding_provider_helper() -> None:
    provider = get_embedding_provider()
    assert provider is not None
    assert hasattr(provider, "embed_texts")

@pytest.mark.asyncio
async def test_mock_provider_l2_normalization_and_dimension() -> None:
    provider = MockEmbeddingProvider(dimension=256)
    assert provider.dimension == 256

    sample_texts = ["EchoMind stabilization phase", "Clean architecture test"]
    embeddings = await provider.embed_texts(sample_texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 256
    assert len(embeddings[1]) == 256

    # Verify L2 Unit Length Normalization
    norm0 = np.linalg.norm(embeddings[0])
    norm1 = np.linalg.norm(embeddings[1])
    assert abs(norm0 - 1.0) < 1e-4
    assert abs(norm1 - 1.0) < 1e-4

def test_qwen_provider_strict_exception_handling() -> None:
    # Attempting to load a non-existent HuggingFace model in QwenEmbeddingProvider MUST raise EchoMindException
    from app.embeddings.qwen_provider import QwenEmbeddingProvider

    with pytest.raises(EchoMindException) as exc_info:
        _ = QwenEmbeddingProvider(model_name="non_existent_invalid_qwen_model_xyz_123")

    assert "Failed to initialize QwenEmbeddingProvider" in str(exc_info.value)
