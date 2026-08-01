# Embeddings package initialization with lazy provider access
from typing import Any

from app.embeddings.backup import JSONLBackupWriter
from app.embeddings.base import BaseEmbeddingProvider
from app.embeddings.checkpoint import CheckpointManager
from app.embeddings.factory import EmbeddingFactory, get_embedding_provider
from app.embeddings.mock_provider import MockEmbeddingProvider
from app.embeddings.pipeline import (
    EmbeddingItem,
    GenericEmbeddingPipeline,
    PipelineMetrics,
    SourceType,
)


def __getattr__(name: str) -> Any:
    """Lazy package-level attribute resolution to prevent import-time provider instantiation."""
    if name == "embedding_provider":
        return get_embedding_provider()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseEmbeddingProvider",
    "MockEmbeddingProvider",
    "EmbeddingFactory",
    "get_embedding_provider",
    "CheckpointManager",
    "JSONLBackupWriter",
    "EmbeddingItem",
    "GenericEmbeddingPipeline",
    "PipelineMetrics",
    "SourceType",
]
