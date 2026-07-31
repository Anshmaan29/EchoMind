from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import get_async_session
from app.knowledge.service import KnowledgeService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.timeline.service import TimelineService

async def get_ingestion_service(
    session: AsyncSession = Depends(get_async_session)
) -> IngestionService:
    """Dependency injector producing an IngestionService instance."""
    return IngestionService(db_session=session)

async def get_document_service(
    session: AsyncSession = Depends(get_async_session)
) -> DocumentService:
    """Dependency injector producing a DocumentService instance."""
    return DocumentService(db_session=session)

async def get_knowledge_service(
    session: AsyncSession = Depends(get_async_session)
) -> KnowledgeService:
    """Dependency injector producing a KnowledgeService instance."""
    return KnowledgeService(db_session=session)

async def get_timeline_service(
    session: AsyncSession = Depends(get_async_session)
) -> TimelineService:
    """Dependency injector producing a TimelineService instance."""
    return TimelineService(db_session=session)
