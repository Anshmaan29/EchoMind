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
    """

    @staticmethod
    def get_provider(provider_name: str = None, dimension: int = None) -> BaseEmbeddingProvider:
        provider_type = (provider_name or settings.EMBEDDING_PROVIDER).lower()
        target_dim = dimension or settings.EMBEDDING_DIMENSION

        if provider_type in ["qwen", "qwen3", "qwen-embedding"]:
            from app.embeddings.qwen_provider import QwenEmbeddingProvider
            return QwenEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL_NAME,
                dimension=target_dim
            )
        elif provider_type in ["bge-m3", "bge"]:
            from app.embeddings.bge_provider import BGEEmbeddingProvider
            return BGEEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL_NAME,
                dimension=target_dim
            )
        elif provider_type == "openai" and settings.OPENAI_API_KEY:
            from app.embeddings.openai_provider import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL_NAME,
                dimension=target_dim
            )

        # Default fallback: Zero-download Mock Embedding Provider
        from app.embeddings.mock_provider import MockEmbeddingProvider
        return MockEmbeddingProvider(dimension=target_dim)

def get_embedding_provider() -> BaseEmbeddingProvider:
    """Utility helper function returning configured embedding provider."""
    return EmbeddingFactory.get_provider()

# Lazy instance property
embedding_provider: BaseEmbeddingProvider = EmbeddingFactory.get_provider()
