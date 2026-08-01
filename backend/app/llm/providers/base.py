from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.search_service import SearchResult

class BaseLLMProvider(ABC):
    """
    Abstract Base Class defining the contract for LLM Providers in EchoMind RAG pipeline.
    """

    @abstractmethod
    async def generate_answer(
        self,
        query: str,
        context_prompt: str,
        results: list[Any] = None
    ) -> str:
        """
        Generates a natural language answer given a user query and structured context.

        :param query: Natural language question.
        :param context_prompt: Assembled prompt from PromptBuilder.
        :param results: Optional list of retrieved SearchResult chunks.
        :return: Generated answer string.
        """
        pass
