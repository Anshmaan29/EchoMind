from app.core.config import settings
from app.embeddings.base import BaseEmbeddingProvider
from app.embeddings.bge_provider import BGEEmbeddingProvider
from app.embeddings.mock_provider import MockEmbeddingProvider
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.embeddings.qwen_provider import QwenEmbeddingProvider

class EmbeddingFactory:
    """Factory class providing embedding provider instances based on system config."""
    
    @staticmethod
    def get_provider() -> BaseEmbeddingProvider:
        provider_type = settings.EMBEDDING_PROVIDER.lower()
        
        if provider_type in ["qwen", "qwen3", "qwen-embedding"]:
            return QwenEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL_NAME,
                dimension=settings.EMBEDDING_DIMENSION
            )
        elif provider_type in ["bge-m3", "bge"]:
            return BGEEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL_NAME,
                dimension=settings.EMBEDDING_DIMENSION
            )
        elif provider_type == "openai" and settings.OPENAI_API_KEY:
            return OpenAIEmbeddingProvider(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL_NAME,
                dimension=settings.EMBEDDING_DIMENSION
            )
        
        # Default fallback to Mock Embedding Provider
        return MockEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)

embedding_provider: BaseEmbeddingProvider = EmbeddingFactory.get_provider()
