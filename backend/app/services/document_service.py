from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import DocumentNotFoundError
from app.models.document import Document
from app.schemas.common import PaginatedResponse
from app.schemas.document import DocumentDetailResponse, DocumentResponse

class DocumentService:
    """
    Application Service managing Document queries, pagination, and deletion transactions.
    """
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def list_documents(self, page: int = 1, page_size: int = 20) -> PaginatedResponse[DocumentResponse]:
        offset = (page - 1) * page_size

        # Count query
        count_stmt = select(func.count(Document.id))
        total_res = await self.db_session.execute(count_stmt)
        total = total_res.scalar() or 0

        # Items query
        stmt = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self.db_session.execute(stmt)
        documents = result.scalars().all()

        items = [DocumentResponse.model_validate(doc) for doc in documents]
        return PaginatedResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size
        )

    async def get_document_by_id(self, document_id: str) -> DocumentDetailResponse:
        stmt = select(Document).where(Document.id == document_id)
        result = await self.db_session.execute(stmt)
        document = result.scalar_one_or_none()

        if not document:
            raise DocumentNotFoundError(document_id=document_id)

        return DocumentDetailResponse.model_validate(document)
