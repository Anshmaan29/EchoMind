# Ingestion package initialization
from app.ingestion.base import IngestionPipeline, PipelineResult
from app.ingestion.chunkers import BaseChunker, ChunkData, TextChunker
from app.ingestion.github_connector import GitHubConnector
from app.ingestion.loaders import BaseLoader, GitHubLoader, ImageLoader, PDFLoader, RawDocumentData
from app.ingestion.parsers import BaseParser, ParsedDocumentData, PDFParser

__all__ = [
    "IngestionPipeline",
    "PipelineResult",
    "BaseLoader",
    "PDFLoader",
    "ImageLoader",
    "GitHubLoader",
    "RawDocumentData",
    "BaseParser",
    "PDFParser",
    "ParsedDocumentData",
    "BaseChunker",
    "TextChunker",
    "ChunkData",
    "GitHubConnector",
]
