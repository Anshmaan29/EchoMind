# Embeddings package initialization with lazy provider access
from app.embeddings.backup import JSONLBackupWriter
from app.embeddings.base import BaseEmbeddingProvider
from app.embeddings.checkpoint import CheckpointManager
from app.embeddings.factory import EmbeddingFactory, embedding_provider, get_embedding_provider
from app.embeddings.mock_provider import MockEmbeddingProvider
from app.embeddings.pipeline import (
    EmbeddingItem,
    GenericEmbeddingPipeline,
    PipelineMetrics,
    SourceType,
)

__all__ = [
    "BaseEmbeddingProvider",
    "MockEmbeddingProvider",
    "EmbeddingFactory",
    "embedding_provider",
    "get_embedding_provider",
    "CheckpointManager",
    "JSONLBackupWriter",
    "EmbeddingItem",
    "GenericEmbeddingPipeline",
    "PipelineMetrics",
    "SourceType",
]
