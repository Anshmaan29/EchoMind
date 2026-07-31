from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel
from app.ingestion.loaders.base import RawDocumentData

class ParsedDocumentData(BaseModel):
    title: str
    clean_text: str
    page_count: int = 1
    extracted_metadata: dict[str, Any] = {}

class BaseParser(ABC):
    """
    Abstract Base Class for Document Parsers in EchoMind.
    Converts raw bytes/binary data into cleaned structured text and metadata.
    """

    @abstractmethod
    async def parse(self, raw_data: RawDocumentData) -> ParsedDocumentData:
        """
        Parses raw document payload into structured clean text.
        
        :param raw_data: RawDocumentData output from BaseLoader.
        :return: ParsedDocumentData container.
        """
        pass
