from abc import ABC, abstractmethod

class BaseEmbeddingProvider(ABC):
    """
    Abstract Base Class defining the contract for all Embedding Providers in EchoMind.
    Enables pluggable integration of OpenAI, BGE, Nomic, Qwen, and Sentence Transformers.
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector dimension length."""
        pass

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generates vector embeddings for a list of string passages.
        
        :param texts: List of text strings to embed.
        :return: List of float vectors.
        """
        pass

    async def embed_single(self, text: str) -> list[float]:
        """Convenience method for embedding a single text string."""
        results = await self.embed_texts([text])
        return results[0] if results else []
