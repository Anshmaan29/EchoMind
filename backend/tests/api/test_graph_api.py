import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_entities_api(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/entities")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_relationships_api(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/relationships")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_graph_search_api(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/graph/search?q=EchoMind")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "subgraph" in data

@pytest.mark.asyncio
async def test_graph_neighbors_api(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/graph/neighbors/non_existent")
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert "relationships" in data
