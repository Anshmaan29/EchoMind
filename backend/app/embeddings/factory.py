from typing import TYPE_CHECKING
from app.core.config import settings
from app.embeddings.base import BaseEmbeddingProvider

if TYPE_CHECKING:
    pass

class EmbeddingFactory:
    """
    Factory class providing embedding provider instances based on system configuration.
    Implements lazy imports so loading `app.embeddings` never triggers heavy PyTorch
    or Transformers downloads during startup.

    Each provider owns its native dimension (Mock: 384, Qwen: 4096, OpenAI: 1536, BGE: 1024).
    Dimension overrides are only passed if explicitly requested by the caller.
    """

    @staticmethod
    def get_provider(provider_name: str | None = None, dimension: int | None = None) -> BaseEmbeddingProvider:
        provider_type = (provider_name or settings.EMBEDDING_PROVIDER).lower()
        dim_kwargs = {"dimension": dimension} if dimension is not None else {}

        if provider_type in ["qwen", "qwen3", "qwen-embedding"]:
            from app.embeddings.qwen_provider import QwenEmbeddingProvider
            return QwenEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL_NAME,
                **dim_kwargs,
            )
        elif provider_type in ["bge-m3", "bge", "sentence_transformers"]:
            from app.embeddings.bge_provider import BGEEmbeddingProvider
            return BGEEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL_NAME,
                **dim_kwargs,
            )
        elif provider_type == "openai":
            from app.embeddings.openai_provider import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL_NAME,
                **dim_kwargs,
            )

        # Default fallback: Zero-download Mock Embedding Provider
        from app.embeddings.mock_provider import MockEmbeddingProvider
        mock_dim = dimension if dimension is not None else settings.EMBEDDING_DIMENSION
        return MockEmbeddingProvider(dimension=mock_dim)


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Utility helper function returning configured embedding provider."""
    return EmbeddingFactory.get_provider()

# Lazy instance property
embedding_provider: BaseEmbeddingProvider = EmbeddingFactory.get_provider()
