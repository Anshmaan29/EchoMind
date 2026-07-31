import hashlib
import re
import numpy as np
from app.embeddings.base import BaseEmbeddingProvider

class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic Semantic Mock Embedding Provider for local development & testing.
    Uses subword n-gram feature hashing, CamelCase/snake_case symbol extraction,
    and term frequency projections to provide semantic search relevance with zero model downloads.
    """
    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _tokenize(self, text: str) -> list[tuple[str, float]]:
        """Extracts words, code symbols, subword n-grams, and assigns TF weights."""
        tokens: list[tuple[str, float]] = []
        if not text.strip():
            return [("empty", 1.0)]

        # 1. Exact raw words (lowercased)
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
        for w in words:
            tokens.append((w, 1.0))
            # Subword 3-grams for fuzzy matching
            if len(w) >= 4:
                for i in range(len(w) - 2):
                    tokens.append((f"ngram_{w[i:i+3]}", 0.4))

        # 2. Extract code identifiers and class names (e.g. QwenEmbeddingProvider, SearchService)
        code_symbols = re.findall(r"\b[A-Za-z0-9_]{3,}\b", text)
        for sym in code_symbols:
            tokens.append((f"sym_{sym.lower()}", 2.5))
            # Split CamelCase (e.g. QwenEmbeddingProvider -> qwen, embedding, provider)
            parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)", sym)
            for p in parts:
                if len(p) >= 3:
                    tokens.append((f"part_{p.lower()}", 1.5))

        return tokens

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self._dimension, dtype=np.float32)
            tokens = self._tokenize(text)

            for token_str, weight in tokens:
                # Deterministic multi-hash feature projection
                h1 = int(hashlib.sha256(token_str.encode("utf-8")).hexdigest(), 16)
                h2 = int(hashlib.md5(token_str.encode("utf-8")).hexdigest(), 16)

                pos1 = h1 % self._dimension
                pos2 = h2 % self._dimension
                sign1 = 1.0 if ((h1 >> 16) & 1) == 1 else -1.0
                sign2 = 1.0 if ((h2 >> 16) & 1) == 1 else -1.0

                vec[pos1] += weight * sign1
                vec[pos2] += (weight * 0.5) * sign2

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())

        return embeddings
