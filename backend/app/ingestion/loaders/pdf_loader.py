import os
from typing import Any
from app.core.exceptions import IngestionException
from app.ingestion.loaders.base import BaseLoader, RawDocumentData

class PDFLoader(BaseLoader):
    """PDF Document Loader implementation for reading raw PDF bytes or file paths."""

    async def load(self, source: Any, title: str | None = None) -> RawDocumentData:
        if isinstance(source, bytes):
            content_bytes = source
            doc_title = title or "uploaded_document.pdf"
        elif isinstance(source, str) and os.path.exists(source):
            with open(source, "rb") as f:
                content_bytes = f.read()
            doc_title = title or os.path.basename(source)
        else:
            raise IngestionException("Invalid PDF source. Expected raw bytes or existing file path.")

        return RawDocumentData(
            title=doc_title,
            content_bytes=content_bytes,
            mime_type="application/pdf",
            source_uri=source if isinstance(source, str) else None,
            meta_data={"size_bytes": len(content_bytes)}
        )
