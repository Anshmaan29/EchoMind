import os
from typing import Any
from app.ingestion.loaders.base import BaseLoader, RawDocumentData

class ImageLoader(BaseLoader):
    """
    Image Data Loader for processing image captions and OCR text extractions.
    """
    async def load(self, source: Any, title: str | None = None) -> RawDocumentData:
        if isinstance(source, bytes):
            content_bytes = source
            doc_title = title or "image_upload.png"
        elif isinstance(source, str) and os.path.exists(source):
            with open(source, "rb") as f:
                content_bytes = f.read()
            doc_title = title or os.path.basename(source)
        else:
            content_bytes = str(source).encode("utf-8")
            doc_title = title or "image_caption_text"

        return RawDocumentData(
            title=doc_title,
            content_bytes=content_bytes,
            mime_type="image/png",
            source_uri=source if isinstance(source, str) else None,
            meta_data={"size_bytes": len(content_bytes), "source_type": "image"}
        )
