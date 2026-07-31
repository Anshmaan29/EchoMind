# Loaders package initialization
from app.ingestion.loaders.base import BaseLoader, RawDocumentData
from app.ingestion.loaders.github_loader import GitHubLoader
from app.ingestion.loaders.image_loader import ImageLoader
from app.ingestion.loaders.pdf_loader import PDFLoader

__all__ = ["BaseLoader", "RawDocumentData", "PDFLoader", "ImageLoader", "GitHubLoader"]
