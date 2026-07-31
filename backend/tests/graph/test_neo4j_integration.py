import pytest
from app.graph.neo4j_store import Neo4jGraphStore
from app.schemas.entity import EntityResponse
from app.schemas.relationship import RelationshipResponse

@pytest.mark.asyncio
async def test_neo4j_graph_store_operations() -> None:
    store = Neo4jGraphStore()
    await store.initialize()

    # Create Entities
    e1 = EntityResponse(id="e1", name="EchoMind", type="Project")
    e2 = EntityResponse(id="e2", name="Neo4j", type="Technology")

    await store.create_entity(e1)
    await store.create_entity(e2)

    found_e1 = await store.find_entity("e1")
    assert found_e1 is not None
    assert found_e1.name == "EchoMind"

    # Create Relationship
    r1 = RelationshipResponse(id="r1", source_id="e1", target_id="e2", relation_type="USES")
    await store.create_relationship(r1)

    neighbors = await store.get_neighbors(entity_id="e1", depth=1)
    assert len(neighbors) == 1
    assert neighbors[0].relation_type == "USES"

    # Search Entities
    results = await store.search_entities(query="EchoMind")
    assert len(results) >= 1
    assert results[0].id == "e1"
