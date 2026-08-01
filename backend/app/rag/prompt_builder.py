from typing import Any
from app.services.search_service import SearchResult


class PromptBuilder:
    """
    RAG Prompt Assembly Engine.
    Assembles retrieved code, document, and note chunks into structured, high-precision
    prompts for LLM synthesis with explicit system instructions, evidence citations, and boundaries.
    """

    @staticmethod
    def build_context_prompt(query: str, results: list[SearchResult]) -> str:
        """
        Formats retrieved SearchResult items into a structured LLM prompt context string.

        :param query: Natural language user question.
        :param results: Retrieved top-k SearchResult chunks.
        :return: Formatted context prompt string with System, Context, Evidence, and Instructions.
        """
        if not results:
            return (
                "=== SYSTEM ===\n"
                "You are EchoMind AI assistant, an expert software developer and digital memory agent.\n\n"
                "=== USER QUESTION ===\n"
                f"{query}\n\n"
                "=== RETRIEVED CONTEXT ===\n"
                "No relevant codebase, document, or personal note chunks were found.\n\n"
                "=== INSTRUCTIONS ===\n"
                "Answer the user question stating clearly that insufficient evidence / context was found in the digital memory index."
            )

        context_blocks = []
        evidence_summary_lines = []

        for idx, res in enumerate(results, start=1):
            source = getattr(res, "source", "code")
            fp = getattr(res, "filepath", "unknown")
            s_line = getattr(res, "start_line", 1)
            e_line = getattr(res, "end_line", 1)
            score = getattr(res, "score", 0.0)
            content = getattr(res, "content", "").strip()

            meta = getattr(res, "meta_data", {}) or {}
            title = meta.get("title") or getattr(res, "filename", fp)
            tags = meta.get("tags", [])
            tags_str = f" | Tags: {', '.join(tags)}" if tags else ""

            block = (
                f"--- [Chunk {idx}] Source: {source} | File: {fp} (L{s_line}-L{e_line}) | Score: {score:.4f}{tags_str} ---\n"
                f"{content}\n"
            )
            context_blocks.append(block)
            evidence_summary_lines.append(f"  [{idx}] {fp} (L{s_line}-L{e_line}) - {title}")

        formatted_context = "\n".join(context_blocks)
        formatted_evidence = "\n".join(evidence_summary_lines)

        prompt = (
            "=== SYSTEM ===\n"
            "You are EchoMind AI assistant, an expert software developer and digital memory agent.\n"
            "Your objective is to provide precise, accurate answers strictly grounded in the retrieved context.\n\n"
            "=== CONVERSATION ===\n"
            f"User Question: {query}\n\n"
            "=== RETRIEVED CONTEXT ===\n"
            f"{formatted_context}\n"
            "=== EVIDENCE SUMMARY ===\n"
            f"{formatted_evidence}\n\n"
            "=== INSTRUCTIONS ===\n"
            "1. Answer ONLY using the retrieved context provided above.\n"
            "2. Cite the specific filepaths and line numbers used in your answer.\n"
            "3. If the retrieved evidence is insufficient or missing key details to answer the question accurately, "
            "state clearly that the evidence is insufficient."
        )

        return prompt
