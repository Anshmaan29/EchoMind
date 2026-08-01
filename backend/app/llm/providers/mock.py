from typing import Any
from app.llm.providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Zero-API Key Mock LLM Provider for EchoMind.
    Synthesizes clean, deterministic natural language answers directly from retrieved codebase
    and note chunks without calling any external LLM APIs or downloading models.
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
                "Please run 'python -m app.cli.embed --input ..' or 'python -m app.cli.notes_index' to index your repository first."
            )

        top_result = results[0]
        top_source = getattr(top_result, "source", "")
        top_meta = getattr(top_result, "meta_data", {}) or {}

        if top_source == "notes" or "title" in top_meta or "tags" in top_meta:
            note_lines = []
            for res in results:
                m = getattr(res, "meta_data", {}) or {}
                t = m.get("title") or getattr(res, "filename", "Untitled Note")
                fp = getattr(res, "filepath", "unknown")
                tg = m.get("tags", [])
                tags_str = f" [Tags: {', '.join(tg)}]" if tg else ""
                line = f"  • Note: '{t}' ({fp}){tags_str}"
                if line not in note_lines:
                    note_lines.append(line)

            t_title = top_meta.get("title") or getattr(top_result, "filename", "Note")
            filepath = getattr(top_result, "filepath", "unknown")
            content = getattr(top_result, "content", "").strip()
            first_line = content.splitlines()[0] if content else ""
            if len(first_line) > 100:
                first_line = first_line[:100] + "..."

            answer_lines = [
                f"Matching notes found in your EchoMind knowledge base:",
                "\n".join(note_lines),
                "",
                f"Primary excerpt from '{t_title}' ({filepath}):",
                f"  {first_line}"
            ]
            return "\n".join(answer_lines)

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
