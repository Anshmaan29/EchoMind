"""
EchoMind Embedding Factory — Lazy Initialization Engine

Provides embedding provider instances based on system configuration.
Implements lazy imports and lazy initialization so loading `app.embeddings`
never triggers heavy PyTorch, Transformers, or model downloads during module import.
The model is loaded on-demand and cached once per process.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
from app.core.config import settings
from app.embeddings.base import BaseEmbeddingProvider

if TYPE_CHECKING:
    pass


class EmbeddingFactory:
    """
    Factory class providing embedding provider instances based on system configuration.
    Each provider owns its native dimension (Mock: 384, Qwen: 4096, OpenAI: 1536, BGE: 1024).
    Dimension overrides are only passed if explicitly requested by the caller.

    Instances are cached at process level so heavy models (like Qwen) are loaded only once.
    """

    _instance_cache: dict[tuple[str, int | None], BaseEmbeddingProvider] = {}

    @classmethod
    def get_provider(
        cls,
        provider_name: str | None = None,
        dimension: int | None = None,
    ) -> BaseEmbeddingProvider:
        p_name = (provider_name or settings.EMBEDDING_PROVIDER).lower().strip()
        cache_key = (p_name, dimension)

        if cache_key in cls._instance_cache:
            return cls._instance_cache[cache_key]

        dim_kwargs = {"dimension": dimension} if dimension is not None else {}

        if p_name in ["qwen", "qwen3", "qwen-embedding"]:
            from app.embeddings.qwen_provider import QwenEmbeddingProvider

            provider = QwenEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL_NAME,
                **dim_kwargs,
            )
        elif p_name in ["bge-m3", "bge", "sentence_transformers"]:
            from app.embeddings.bge_provider import BGEEmbeddingProvider

            provider = BGEEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL_NAME,
                **dim_kwargs,
            )
        elif p_name == "openai":
            from app.embeddings.openai_provider import OpenAIEmbeddingProvider

            provider = OpenAIEmbeddingProvider(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL_NAME,
                **dim_kwargs,
            )
        else:
            # Default fallback: Zero-download Mock Embedding Provider
            from app.embeddings.mock_provider import MockEmbeddingProvider

            mock_dim = dimension if dimension is not None else settings.EMBEDDING_DIMENSION
            provider = MockEmbeddingProvider(dimension=mock_dim)

        cls._instance_cache[cache_key] = provider
        return provider

    @classmethod
    def clear_cache(cls) -> None:
        """Clears process-level provider instance cache (useful for tests)."""
        cls._instance_cache.clear()


def get_embedding_provider(
    provider_name: str | None = None,
    dimension: int | None = None,
) -> BaseEmbeddingProvider:
    """Utility helper function returning configured embedding provider (lazy loaded)."""
    return EmbeddingFactory.get_provider(provider_name=provider_name, dimension=dimension)


def __getattr__(name: str) -> Any:
    """Lazy module-level attribute resolution to prevent import-time provider instantiation."""
    if name == "embedding_provider":
        return get_embedding_provider()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
