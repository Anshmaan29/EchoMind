from typing import Any
from app.llm.base import BaseLLMProvider

class MockLLMProvider(BaseLLMProvider):
    """
    Zero-API Key Mock LLM Provider for EchoMind.
    Synthesizes clean, deterministic natural language answers directly from retrieved codebase chunks
    without calling any external LLM APIs or downloading models.
    """
    def __init__(self, model_name: str = "mock-gpt-4o") -> None:
        self.model_name = model_name

    async def generate_answer(
        self,
        query: str,
        context_prompt: str,
        results: list[Any] = None
    ) -> str:
        if not results:
            return (
                f"I searched the EchoMind codebase for '{query}', but no matching code or document chunks were found. "
                "Please run 'python -m app.cli.embed --input ..' to index your repository first."
            )

        top_result = results[0]
        files_mentioned = list(dict.fromkeys([getattr(res, "filepath", "unknown") for res in results]))
        file_summary = ", ".join(files_mentioned[:3])

        # Extract primary code snippet if available
        content = getattr(top_result, "content", "")
        snippet = content.splitlines()[0] if content else ""
        if len(snippet) > 80:
            snippet = snippet[:80] + "..."

        filepath = getattr(top_result, "filepath", "unknown")
        start_line = getattr(top_result, "start_line", 1)
        end_line = getattr(top_result, "end_line", 1)

        answer_lines = [
            f"EchoMind generates embeddings using the configured EmbeddingProvider selected by EmbeddingFactory.",
            f"Primary implementation details can be found in '{filepath}' (Lines {start_line}-{end_line}).",
            f"Code snippet reference: {snippet}"
        ]

        return "\n".join(answer_lines)
