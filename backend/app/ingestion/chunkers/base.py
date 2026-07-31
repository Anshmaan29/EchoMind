from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel
from app.ingestion.parsers.base import ParsedDocumentData

class ChunkData(BaseModel):
    chunk_index: int
    content: str
    token_count: int
    meta_data: dict[str, Any] = {}

class BaseChunker(ABC):
    """
    Abstract Base Class for Text Chunkers in EchoMind.
    Segments parsed text into semantic windows for embedding generation.
    """

    @abstractmethod
    async def chunk(self, parsed_data: ParsedDocumentData) -> list[ChunkData]:
        """
        Segments parsed document into a list of ChunkData instances.
        
        :param parsed_data: ParsedDocumentData container.
        :return: List of ChunkData instances.
        """
        pass
