from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from app.llm.base import BaseLLMProvider
from app.rag.prompt_builder import PromptBuilder
from app.services.search_service import SearchResult, SearchService

class EvidenceItem(BaseModel):
    """
    Structured Evidence Citation Model preserving file paths, line ranges, and relevance scores.
    """
    model_config = ConfigDict(from_attributes=True)

    filepath: str
    filename: str
    start_line: int
    end_line: int
    score: float
    source: str

class AskResponse(BaseModel):
    """
    End-to-End RAG Answer Response with structured evidence citations.
    """
    model_config = ConfigDict(from_attributes=True)

    question: str
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)

class AskService:
    """
    EchoMind RAG Pipeline Orchestrator.
    Pipeline: Question -> SearchService -> PromptBuilder -> LLMProvider -> AskResponse.
    """
    def __init__(
        self,
        search_service: SearchService = None,
        llm_provider_inst: BaseLLMProvider = None
    ) -> None:
        self.search_service = search_service or SearchService()
        if llm_provider_inst is not None:
            self.llm = llm_provider_inst
        else:
            from app.llm.factory import get_llm_provider
            self.llm = get_llm_provider()

    async def ask(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> AskResponse:
        """
        Executes end-to-end RAG question answering.
        
        :param question: Natural language user question.
        :param top_k: Number of context chunks to retrieve.
        :param min_score: Minimum similarity score threshold.
        :return: AskResponse object containing answer and evidence citations.
        """
        # 1. Retrieve top-k context chunks via SearchService
        search_results: list[SearchResult] = await self.search_service.search(
            query=question,
            top_k=top_k,
            min_score=min_score
        )

        # 2. Build structured context prompt
        context_prompt = PromptBuilder.build_context_prompt(
            query=question,
            results=search_results
        )

        # 3. Generate answer via configured LLM Provider
        generated_answer = await self.llm.generate_answer(
            query=question,
            context_prompt=context_prompt,
            results=search_results
        )

        # 4. Assemble Evidence Citations
        evidence_items = [
            EvidenceItem(
                filepath=res.filepath,
                filename=res.filename,
                start_line=res.start_line,
                end_line=res.end_line,
                score=res.score,
                source=res.source
            )
            for res in search_results
        ]

        return AskResponse(
            question=question,
            answer=generated_answer,
            evidence=evidence_items
        )
