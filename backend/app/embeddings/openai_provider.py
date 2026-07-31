import httpx
from app.core.exceptions import EchoMindException
from app.embeddings.base import BaseEmbeddingProvider

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI REST Embedding Provider supporting text-embedding-3-small, text-embedding-3-large.
    """
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small", dimension: int = 1536) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise EchoMindException("OpenAI API Key is required for OpenAIEmbeddingProvider.")

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model_name,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise EchoMindException(f"OpenAI Embedding API error ({response.status_code}): {response.text}")
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
