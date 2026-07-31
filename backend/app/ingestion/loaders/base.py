from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class RawDocumentData(BaseModel):
    title: str
    content_bytes: bytes
    mime_type: str
    source_uri: str | None = None
    meta_data: dict[str, Any] = {}

class BaseLoader(ABC):
    """
    Abstract Base Class for Document Loaders in EchoMind.
    Reused across PDF, GitHub, WhatsApp, Audio, Images, and Email connectors.
    """

    @abstractmethod
    async def load(self, source: Any, title: str | None = None) -> RawDocumentData:
        """
        Loads raw payload data from input source.
        
        :param source: File path, byte stream, or URL target.
        :param title: Optional document title.
        :return: RawDocumentData container.
        """
        pass
