from app.vector.base import BaseVectorStore
from app.vector.qdrant_store import QdrantVectorStore

class VectorStoreFactory:
    """Factory class providing VectorStore instances based on system configuration."""
    
    @staticmethod
    def get_vector_store() -> BaseVectorStore:
        return QdrantVectorStore()

vector_store: BaseVectorStore = VectorStoreFactory.get_vector_store()
