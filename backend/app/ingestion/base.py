from typing import Any
from pydantic import BaseModel
from app.embeddings.base import BaseEmbeddingProvider
from app.ingestion.chunkers.base import BaseChunker, ChunkData
from app.ingestion.loaders.base import BaseLoader, RawDocumentData
from app.ingestion.parsers.base import BaseParser, ParsedDocumentData

class PipelineResult(BaseModel):
    title: str
    raw_document: RawDocumentData
    parsed_document: ParsedDocumentData
    chunks: list[ChunkData]
    embeddings: list[list[float]]

class IngestionPipeline:
    """
    Composite Ingestion Pipeline binding Loader -> Parser -> Chunker -> Embedder.
    Designed for reuse across future connectors (GitHub, WhatsApp, Audio, Images, Emails).
    """
    def __init__(
        self,
        loader: BaseLoader,
        parser: BaseParser,
        chunker: BaseChunker,
        embedder: BaseEmbeddingProvider
    ) -> None:
        self.loader = loader
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder

    async def run(self, source: Any, title: str | None = None) -> PipelineResult:
        # 1. Load raw payload
        raw_doc = await self.loader.load(source=source, title=title)

        # 2. Parse text & metadata
        parsed_doc = await self.parser.parse(raw_data=raw_doc)

        # 3. Chunk text passages
        chunks = await self.chunker.chunk(parsed_data=parsed_doc)

        # 4. Generate embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = await self.embedder.embed_texts(chunk_texts) if chunk_texts else []

        return PipelineResult(
            title=parsed_doc.title,
            raw_document=raw_doc,
            parsed_document=parsed_doc,
            chunks=chunks,
            embeddings=embeddings
        )
