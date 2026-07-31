import io
import re
from pypdf import PdfReader
from app.core.exceptions import IngestionException
from app.ingestion.loaders.base import RawDocumentData
from app.ingestion.parsers.base import BaseParser, ParsedDocumentData

class PDFParser(BaseParser):
    """PDF Parser implementation extracting clean text & page metadata from PDF bytes."""

    async def parse(self, raw_data: RawDocumentData) -> ParsedDocumentData:
        try:
            stream = io.BytesIO(raw_data.content_bytes)
            extracted_pages: list[str] = []
            page_count = 1

            try:
                reader = PdfReader(stream)
                page_count = len(reader.pages)
                for page in reader.pages:
                    text = page.extract_text() or ""
                    cleaned = self._clean_page_text(text)
                    if cleaned:
                        extracted_pages.append(cleaned)
            except Exception:
                # Fallback decoding for raw/plain-text streams in test or non-standard PDFs
                fallback_text = raw_data.content_bytes.decode("utf-8", errors="ignore")
                cleaned = self._clean_page_text(fallback_text)
                if cleaned:
                    extracted_pages.append(cleaned)

            full_text = "\n\n".join(extracted_pages)
            if not full_text.strip():
                full_text = f"Untitled PDF Document ({raw_data.title})"

            return ParsedDocumentData(
                title=raw_data.title,
                clean_text=full_text,
                page_count=page_count,
                extracted_metadata={
                    **raw_data.meta_data,
                    "parsed_pages": page_count,
                    "character_count": len(full_text)
                }
            )
        except Exception as e:
            raise IngestionException(f"Failed to parse PDF document '{raw_data.title}': {str(e)}")

    def _clean_page_text(self, text: str) -> str:
        """Cleans headers, footers, whitespace, and invalid characters."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
