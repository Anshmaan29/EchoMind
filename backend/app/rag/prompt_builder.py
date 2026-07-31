from typing import Any
from app.services.search_service import SearchResult

class PromptBuilder:
    """
    RAG Prompt Assembly Engine.
    Assembles retrieved code and document chunks into a structured prompt context
    preserving file paths, line numbers, and similarity scores for LLM synthesis.
    """

    @staticmethod
    def build_context_prompt(query: str, results: list[SearchResult]) -> str:
        """
        Formats retrieved SearchResult items into a structured LLM context string.
        
        :param query: Natural language user question.
        :param results: Retrieved top-k SearchResult chunks.
        :return: Formatted context prompt string.
        """
        if not results:
            return (
                f"Question: {query}\n\n"
                "Context:\nNo relevant codebase or document chunks were found.\n\n"
                "Instructions:\nAnswer the question stating that no relevant context was found."
            )

        context_blocks = []
        for idx, res in enumerate(results, start=1):
            block = (
                f"--- Context Chunk [{idx}] ---\n"
                f"Filepath: {res.filepath}\n"
                f"Line Range: L{res.start_line}-L{res.end_line}\n"
                f"Relevance Score: {res.score:.4f}\n"
                f"Source: {res.source}\n"
                f"Content:\n{res.content}\n"
            )
            context_blocks.append(block)

        formatted_context = "\n".join(context_blocks)

        prompt = (
            f"User Question: {query}\n\n"
            "Retrieved Codebase & Document Context:\n"
            f"{formatted_context}\n"
            "Instructions:\n"
            "Answer the user question strictly using the provided context chunks. "
            "Cite the specific filepaths and line ranges in your explanation."
        )

        return prompt
