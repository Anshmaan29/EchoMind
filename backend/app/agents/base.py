from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class AgentQueryRequest(BaseModel):
    query: str
    user_id: str | None = None
    context: dict[str, Any] = {}

class AgentQueryResponse(BaseModel):
    answer: str
    thought_process: list[str] = []
    sources: list[str] = []

class BaseMemoryAgent(ABC):
    """
    Abstract Base Class contract for AI Memory Agents in EchoMind (Milestone 2 integration).
    """

    @abstractmethod
    async def process_query(self, request: AgentQueryRequest) -> AgentQueryResponse:
        """Processes a natural language memory query across vector and graph indexes."""
        pass
