from typing import Any
from app.embeddings.base import BaseEmbeddingProvider
from app.graph.base import BaseGraphStore
from app.vector.base import BaseVectorStore

class MemoryReasoner:
    """
    Hybrid Graph + Vector Reasoning service.
    Fuses dense vector semantic passage retrieval with knowledge graph entity multi-hop neighborhood traversal.
    """
    def __init__(
        self,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
        embedder: BaseEmbeddingProvider
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder

    async def hybrid_memory_search(
        self,
        query: str,
        collection_name: str,
        limit: int = 5
    ) -> dict[str, Any]:
        """
        Executes parallel Vector similarity search and Graph neighborhood discovery.
        """
        # 1. Vector Search
        query_vector = await self.embedder.embed_single(query)
        vector_results = await self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )

        # 2. Graph Search
        matched_entities = await self.graph_store.search_entities(query=query, limit=limit)
        graph_subgraph = []
        for ent in matched_entities:
            neighbors = await self.graph_store.get_neighbors(entity_id=ent.id, depth=1)
            graph_subgraph.extend(neighbors)

        return {
            "query": query,
            "vector_passages": [
                {
                    "score": res.score,
                    "content": res.payload.get("content", ""),
                    "document_id": res.payload.get("document_id", ""),
                }
                for res in vector_results
            ],
            "graph_entities": [ent.model_dump() for ent in matched_entities],
            "graph_relationships": [rel.model_dump() for rel in graph_subgraph],
        }
