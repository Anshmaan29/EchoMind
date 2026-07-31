import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_timeline_api(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/timeline")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert "total_events" in data

@pytest.mark.asyncio
async def test_project_timeline_api(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/timeline/project/EchoMind")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "EchoMind"
    assert "events" in data
