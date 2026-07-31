import hashlib
from app.embeddings.base import BaseEmbeddingProvider

class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic Mock Embedding Provider for testing and keyless local execution.
    Generates normalized pseudo-embeddings via MD5 hashing.
    """
    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dimension
            words = text.lower().split() or ["empty"]
            for idx, word in enumerate(words):
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                pos = h % self._dimension
                val = ((h >> 8) % 100) / 50.0 - 1.0
                vec[pos] += val * (1.0 / ((idx + 1) ** 0.5))
            
            # Normalize vector
            norm = (sum(x * x for x in vec)) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings
