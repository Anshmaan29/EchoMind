import re
from app.ingestion.chunkers.base import BaseChunker, ChunkData
from app.ingestion.parsers.base import ParsedDocumentData

class TextChunker(BaseChunker):
    """
    Recursive Character Text Chunker with overlap and sentence boundary splitting.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, parsed_data: ParsedDocumentData) -> list[ChunkData]:
        text = parsed_data.clean_text
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks: list[ChunkData] = []
        current_buffer = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_buffer) + len(para) + 2 <= self.chunk_size:
                current_buffer = f"{current_buffer}\n\n{para}".strip()
            else:
                if current_buffer:
                    chunks.append(self._create_chunk_data(current_buffer, chunk_idx, parsed_data))
                    chunk_idx += 1
                
                if len(para) > self.chunk_size:
                    # Sentence splitting for long paragraphs
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    sub_buf = ""
                    for sent in sentences:
                        if len(sub_buf) + len(sent) + 1 <= self.chunk_size:
                            sub_buf = f"{sub_buf} {sent}".strip()
                        else:
                            if sub_buf:
                                chunks.append(self._create_chunk_data(sub_buf, chunk_idx, parsed_data))
                                chunk_idx += 1
                            sub_buf = sent
                    current_buffer = sub_buf
                else:
                    current_buffer = para

        if current_buffer:
            chunks.append(self._create_chunk_data(current_buffer, chunk_idx, parsed_data))

        return chunks

    def _create_chunk_data(self, text: str, index: int, parsed_data: ParsedDocumentData) -> ChunkData:
        # Simple word token count estimation
        token_count = len(text.split())
        return ChunkData(
            chunk_index=index,
            content=text,
            token_count=token_count,
            meta_data={
                "title": parsed_data.title,
                "character_length": len(text)
            }
        )
