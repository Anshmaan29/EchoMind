from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from app.api.dependencies import get_document_service, get_ingestion_service
from app.core.exceptions import IngestionException
from app.schemas.common import PaginatedResponse
from app.schemas.document import DocumentDetailResponse, DocumentResponse
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService

router = APIRouter(tags=["Documents"])

@router.post(
    "/upload/pdf",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document"
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF document file to upload and ingest"),
    ingestion_service: IngestionService = Depends(get_ingestion_service)
) -> DocumentResponse:
    """
    Uploads a PDF file, parses text content, generates semantic chunk embeddings,
    stores vectors in Qdrant, and persists document metadata in PostgreSQL.
    """
    filename = file.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF file format (.pdf) is supported by this endpoint."
        )

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        return await ingestion_service.ingest_pdf(file_bytes=content_bytes, filename=filename)
    except IngestionException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing PDF ingestion: {str(e)}"
        )

@router.get(
    "/documents",
    response_model=PaginatedResponse[DocumentResponse],
    summary="List all ingested documents"
)
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page index (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    document_service: DocumentService = Depends(get_document_service)
) -> PaginatedResponse[DocumentResponse]:
    """Retrieves a paginated list of ingested documents and their metadata."""
    return await document_service.list_documents(page=page, page_size=page_size)

@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details by ID"
)
async def get_document_by_id(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentDetailResponse:
    """Retrieves detailed metadata and chunk hierarchy for a document by ID."""
    return await document_service.get_document_by_id(document_id=document_id)
