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


class TimelineAwareAskService(AskService):
    """
    EchoMind Timeline-Aware RAG Orchestrator.

    Extends AskService with temporal intelligence: detects time-based questions
    ("What did I work on today?", "What changed this week?") and enriches the
    LLM context with a chronological timeline before standard RAG retrieval.

    Falls back to the base AskService pipeline for non-temporal questions.

    Usage::

        service = TimelineAwareAskService()
        response = await service.ask("What did I work on today?")
    """

    def __init__(
        self,
        search_service: SearchService = None,
        llm_provider_inst: BaseLLMProvider = None,
        timeline_service: Any = None,
    ) -> None:
        super().__init__(
            search_service=search_service,
            llm_provider_inst=llm_provider_inst,
        )
        # Lazy-init to avoid circular import at module load time
        self._timeline_service = timeline_service

    def _get_timeline_service(self) -> Any:
        if self._timeline_service is None:
            from app.services.timeline_service import TimelineService
            self._timeline_service = TimelineService(
                search_service=self.search_service,
            )
        return self._timeline_service

    async def ask(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> AskResponse:
        """
        Executes RAG with optional timeline enrichment.

        For temporal questions ("What did I work on today?", "What changed this
        week?") the answer is formatted **directly from TimelineService data**
        rather than being delegated to the LLM.  This guarantees that timeline
        questions always receive a timeline answer regardless of which LLM
        provider is configured.

        Flow for temporal questions:
          1. Fetch timeline events (once — reused for summary).
          2. If no events: fall back to standard RAG.
          3. Format a structured answer from commits + files + modules.
          4. Use git commit paths as evidence citations.

        Non-temporal questions delegate to the base AskService pipeline.

        :param question:  Natural language user question.
        :param top_k:     Number of retrieved context chunks (default: 5).
        :param min_score: Minimum similarity score threshold (default: 0.0).
        :return: AskResponse with answer and evidence citations.
        """
        from app.services.timeline_service import detect_temporal_intent

        period = detect_temporal_intent(question)

        if period is None:
            # No temporal intent — delegate to standard RAG pipeline
            return await super().ask(question=question, top_k=top_k, min_score=min_score)

        # ------------------------------------------------------------------ #
        # Temporal path
        # ------------------------------------------------------------------ #
        timeline_svc = self._get_timeline_service()

        try:
            # 1. Fetch events once
            events = await timeline_svc.get_events_for_period(period=period)

            if not events:
                # No timeline data — fall back to standard RAG
                return await super().ask(question=question, top_k=top_k, min_score=min_score)

            # 2. Build summary from the SAME events (no second pipeline run)
            summary = await timeline_svc.build_daily_summary(events=events)

            # 3. Format answer directly from timeline data
            answer = self._format_timeline_answer(events, summary, period)

            # 4. Build evidence from git commits + changed files
            evidence_items = self._build_timeline_evidence(events, summary)

        except Exception:
            # Graceful degradation: fall back to pure RAG if timeline fails
            return await super().ask(question=question, top_k=top_k, min_score=min_score)

        return AskResponse(
            question=question,
            answer=answer,
            evidence=evidence_items,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_timeline_answer(
        events: list,
        summary: Any,
        period: str,
    ) -> str:
        """
        Formats a structured plain-text answer from timeline events + summary.
        This bypasses the LLM so the answer is always timeline-driven.
        """
        import os as _os

        period_labels = {
            "today": "Today's Work",
            "week":  "This Week's Work",
            "month": "This Month's Work",
        }
        heading = period_labels.get(period, "Work Summary")

        lines: list[str] = [heading, ""]

        # --- Commits ---
        git_events = [e for e in events if e.source == "git"]
        if git_events:
            lines.append("Commits:")
            for ev in git_events[:10]:
                short = ev.metadata.get("short_hash", "")
                prefix = f"  [{short}]  " if short else "  "
                lines.append(f"{prefix}• {ev.title}")
            lines.append("")

        # --- Files changed ---
        if summary.files_changed:
            lines.append("Files changed:")
            for fp in summary.files_changed[:15]:
                lines.append(f"  • {fp}")
            if len(summary.files_changed) > 15:
                lines.append(f"  ... and {len(summary.files_changed) - 15} more")
            lines.append("")

        # --- Modules ---
        if summary.major_modules:
            lines.append("Major modules touched:")
            for m in summary.major_modules[:10]:
                lines.append(f"  • {m}")
            lines.append("")

        # --- New classes / functions ---
        if summary.new_classes:
            lines.append("New classes / functions:")
            for c in summary.new_classes[:8]:
                lines.append(f"  • {c}")
            lines.append("")

        # --- Documentation ---
        if summary.documentation_updates:
            lines.append("Documentation updates:")
            for doc in summary.documentation_updates[:5]:
                lines.append(f"  • {_os.path.basename(doc)}")
            lines.append("")

        # --- Fallback if everything is empty ---
        if not git_events and not summary.files_changed:
            code_events = [e for e in events if e.source == "code"]
            md_events   = [e for e in events if e.source in ("markdown", "doc")]
            if code_events or md_events:
                lines.append("Recent activity (code & docs):")
                for ev in (code_events + md_events)[:10]:
                    lines.append(f"  • {ev.title}")
            else:
                lines.append("No recent activity found.")

        return "\n".join(lines).rstrip()

    @staticmethod
    def _build_timeline_evidence(events: list, summary: Any) -> list:
        """
        Builds evidence citations from git commits and changed files.
        These are the actual sources for a timeline answer — NOT code chunks.
        """
        import os as _os
        evidence: list[EvidenceItem] = []

        # Git commits as primary evidence
        for ev in events:
            if ev.source == "git":
                short = ev.metadata.get("short_hash", "commit")
                evidence.append(EvidenceItem(
                    filepath=ev.filepath,
                    filename=f"commit_{short}",
                    start_line=1,
                    end_line=1,
                    score=1.0,
                    source="git",
                ))

        # Changed files as secondary evidence
        for fp in summary.files_changed[:5]:
            evidence.append(EvidenceItem(
                filepath=fp,
                filename=_os.path.basename(fp),
                start_line=1,
                end_line=1,
                score=0.90,
                source="code",
            ))

        return evidence
