from app.graph.base import BaseGraphStore
from app.graph.neo4j_store import Neo4jGraphStore

class GraphStoreFactory:
    """Factory class providing GraphStore instances based on configuration."""
    
    @staticmethod
    def get_graph_store() -> BaseGraphStore:
        return Neo4jGraphStore()

graph_store: BaseGraphStore = GraphStoreFactory.get_graph_store()
