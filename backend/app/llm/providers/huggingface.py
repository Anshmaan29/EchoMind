from typing import Any
from app.core.exceptions import EchoMindException
from app.llm.providers.base import BaseLLMProvider


class HuggingFaceLLMProvider(BaseLLMProvider):
    """
    Local HuggingFace LLM Provider for EchoMind.

    Architecture specification for running open-weights models (e.g. Qwen2.5-7B, Llama-3-8B).
    Lightweight design: no models downloaded or loaded during initialization.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cpu",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream

    def _load_model(self) -> None:
        """Loads Hugging Face model and tokenizer into memory."""
        raise EchoMindException("Model not initialized.")

    def _tokenize(self, text: str) -> Any:
        """Tokenizes text for input to Hugging Face model."""
        raise EchoMindException("Model not initialized.")

    def _format_prompt(self, query: str, context_prompt: str) -> str:
        """Formats prompt using model chat template."""
        raise EchoMindException("Model not initialized.")

    async def generate_answer(
        self,
        query: str,
        context_prompt: str,
        results: list[Any] = None,
    ) -> str:
        """Generates answer using Hugging Face model inference."""
        raise EchoMindException("Model not initialized.")
