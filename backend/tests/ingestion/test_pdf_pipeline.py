import pytest
from app.embeddings.mock_provider import MockEmbeddingProvider
from app.ingestion.base import IngestionPipeline
from app.ingestion.chunkers.text_chunker import TextChunker
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.loaders.base import RawDocumentData

@pytest.mark.asyncio
async def test_pdf_loader_bytes() -> None:
    loader = PDFLoader()
    raw = await loader.load(b"dummy pdf bytes", title="test.pdf")
    assert raw.title == "test.pdf"
    assert raw.mime_type == "application/pdf"
    assert raw.content_bytes == b"dummy pdf bytes"

@pytest.mark.asyncio
async def test_text_chunker() -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    sample_text = "EchoMind is an AI-powered Digital Memory Operating System. It reconstructs a user's digital history into an intelligent memory graph."
    
    from app.ingestion.parsers.base import ParsedDocumentData
    parsed = ParsedDocumentData(title="Test", clean_text=sample_text)
    
    chunks = await chunker.chunk(parsed)
    assert len(chunks) >= 1
    assert chunks[0].chunk_index == 0
    assert "EchoMind" in chunks[0].content

@pytest.mark.asyncio
async def test_full_pipeline_mock() -> None:
    pipeline = IngestionPipeline(
        loader=PDFLoader(),
        parser=PDFParser(),
        chunker=TextChunker(chunk_size=200),
        embedder=MockEmbeddingProvider(dimension=64)
    )
    
    result = await pipeline.run(b"Sample PDF text content for EchoMind pipeline testing.", title="sample.pdf")
    assert result.title == "sample.pdf"
    assert len(result.chunks) >= 1
    assert len(result.embeddings) == len(result.chunks)
    assert len(result.embeddings[0]) == 64
