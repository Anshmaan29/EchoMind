import uuid
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.factory import get_embedding_provider
from app.extraction.entity_extractor import EntityExtractor
from app.extraction.relation_extractor import RelationshipExtractor
from app.graph.factory import graph_store
from app.ingestion.base import IngestionPipeline
from app.ingestion.chunkers.text_chunker import TextChunker
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.parsers.pdf_parser import PDFParser
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.entity import Entity
from app.models.relationship import Relationship
from app.models.timeline import TimelineEvent
from app.schemas.document import DocumentResponse
from app.schemas.entity import EntityResponse
from app.schemas.relationship import RelationshipResponse
from app.timeline.engine import TimelineEngine
from app.utils.helpers import compute_hash
from app.vector.base import VectorRecord
from app.vector.factory import vector_store

class IngestionService:
    """
    Application Service orchestrating full Milestone 2 Extended Ingestion Pipeline:
    Upload PDF -> Parse -> Chunk -> Entity Extraction -> Relationship Extraction -> Timeline Event Creation
    -> Embeddings -> Qdrant Vector Store -> Knowledge Graph -> PostgreSQL Database.
    """
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session
        self.pipeline = IngestionPipeline(
            loader=PDFLoader(),
            parser=PDFParser(),
            chunker=TextChunker(chunk_size=500, chunk_overlap=100),
            embedder=get_embedding_provider()
        )
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.timeline_engine = TimelineEngine()
        self.graph_store = graph_store

    async def ingest_pdf(self, file_bytes: bytes, filename: str) -> DocumentResponse:
        doc_id = str(uuid.uuid4())
        file_hash = compute_hash(file_bytes)

        logger.info(f"Starting extended PDF ingestion for '{filename}' (id: {doc_id})...")

        # 1. Execute base ingestion pipeline (Load -> Parse -> Chunk -> Embed)
        pipeline_result = await self.pipeline.run(source=file_bytes, title=filename)
        clean_text = pipeline_result.parsed_document.clean_text

        # 2. Extract Entities
        extracted_entities = await self.entity_extractor.extract_entities(
            text=clean_text,
            source_document_id=doc_id
        )

        # 3. Extract Relationships
        extracted_rels = await self.relationship_extractor.extract_relationships(
            text=clean_text,
            entities=extracted_entities,
            source_document_id=doc_id
        )

        # 4. Extract Timeline Events
        extracted_events = await self.timeline_engine.create_timeline_events(
            text=clean_text,
            entities=extracted_entities,
            source_document_id=doc_id
        )

        # 5. Initialize Stores
        await vector_store.initialize_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            dimension=self.pipeline.embedder.dimension
        )
        await self.graph_store.initialize()

        # 6. Save Document ORM
        doc_entity = Document(
            id=doc_id,
            title=pipeline_result.title,
            source_type="pdf",
            file_hash=file_hash,
            file_size_bytes=len(file_bytes),
            status="completed",
            raw_text=clean_text,
            meta_data=pipeline_result.parsed_document.extracted_metadata
        )
        self.db_session.add(doc_entity)

        # 7. Save Chunks & Qdrant Vectors
        chunk_orm_list: list[DocumentChunk] = []
        vector_records: list[VectorRecord] = []

        for chunk_data, embedding_vec in zip(pipeline_result.chunks, pipeline_result.embeddings):
            chunk_id = str(uuid.uuid4())
            vector_id = str(uuid.uuid4())

            chunk_orm_list.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=doc_id,
                    chunk_index=chunk_data.chunk_index,
                    content=chunk_data.content,
                    token_count=chunk_data.token_count,
                    vector_id=vector_id,
                    meta_data=chunk_data.meta_data
                )
            )

            vector_records.append(
                VectorRecord(
                    id=vector_id,
                    vector=embedding_vec,
                    payload={
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_data.chunk_index,
                        "title": pipeline_result.title,
                        "content": chunk_data.content,
                    }
                )
            )

        self.db_session.add_all(chunk_orm_list)

        # 8. Save Entities to DB & Knowledge Graph
        entity_name_to_id: dict[str, str] = {}
        for e_create in extracted_entities:
            e_id = str(uuid.uuid4())
            entity_name_to_id[e_create.name] = e_id

            e_orm = Entity(
                id=e_id,
                name=e_create.name,
                type=e_create.type,
                aliases=e_create.aliases,
                description=e_create.description,
                confidence=e_create.confidence,
                source_document_id=doc_id,
                meta_data=e_create.meta_data
            )
            self.db_session.add(e_orm)

            # Knowledge Graph node
            e_resp = EntityResponse(
                id=e_id,
                name=e_create.name,
                type=e_create.type,
                aliases=e_create.aliases,
                description=e_create.description,
                confidence=e_create.confidence,
                source_document_id=doc_id,
                meta_data=e_create.meta_data
            )
            await self.graph_store.create_entity(e_resp)

        # 9. Save Relationships to DB & Knowledge Graph
        for r_create in extracted_rels:
            r_id = str(uuid.uuid4())
            src_id = entity_name_to_id.get(r_create.source_id, r_create.source_id)
            tgt_id = entity_name_to_id.get(r_create.target_id, r_create.target_id)

            r_orm = Relationship(
                id=r_id,
                source_id=src_id,
                target_id=tgt_id,
                relation_type=r_create.relation_type,
                confidence=r_create.confidence,
                evidence=r_create.evidence,
                source_document_id=doc_id,
                meta_data=r_create.meta_data
            )
            self.db_session.add(r_orm)

            # Knowledge Graph edge
            r_resp = RelationshipResponse(
                id=r_id,
                source_id=src_id,
                target_id=tgt_id,
                relation_type=r_create.relation_type,
                confidence=r_create.confidence,
                evidence=r_create.evidence,
                source_document_id=doc_id,
                meta_data=r_create.meta_data
            )
            await self.graph_store.create_relationship(r_resp)

        # 10. Save Timeline Events
        for ev_create in extracted_events:
            ev_id = str(uuid.uuid4())
            ev_orm = TimelineEvent(
                id=ev_id,
                title=ev_create.title,
                description=ev_create.description,
                timestamp=ev_create.timestamp,
                entities_involved=ev_create.entities_involved,
                projects_involved=ev_create.projects_involved,
                importance_score=ev_create.importance_score,
                source_document_id=doc_id,
                meta_data=ev_create.meta_data
            )
            self.db_session.add(ev_orm)

        await self.db_session.flush()

        # 11. Upsert Qdrant Vectors
        if vector_records:
            await vector_store.upsert_records(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                records=vector_records
            )

        logger.info(
            f"Successfully completed extended ingestion for '{filename}': "
            f"{len(chunk_orm_list)} Chunks, {len(extracted_entities)} Entities, "
            f"{len(extracted_rels)} Relationships, {len(extracted_events)} Timeline Events."
        )

        return DocumentResponse.model_validate(doc_entity)
