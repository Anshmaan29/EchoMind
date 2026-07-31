import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_get_documents_empty(async_client: AsyncClient) -> None:
    response = await async_client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["total"] == 0

@pytest.mark.asyncio
async def test_upload_non_pdf_fails(async_client: AsyncClient) -> None:
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = await async_client.post("/upload/pdf", files=files)
    assert response.status_code == 400
    assert "Only PDF file format (.pdf) is supported" in response.json()["detail"]
