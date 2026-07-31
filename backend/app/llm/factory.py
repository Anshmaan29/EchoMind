from app.core.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.mock_provider import MockLLMProvider

class LLMFactory:
    """
    Factory providing LLM Provider instances based on configuration.
    """

    @staticmethod
    def get_provider(provider_name: str = None) -> BaseLLMProvider:
        provider_type = (provider_name or settings.LLM_PROVIDER).lower()

        if provider_type == "mock":
            return MockLLMProvider(model_name=settings.LLM_MODEL_NAME)
        
        # Future LLM provider stubs (OpenAI, Anthropic, OpenRouter) fall back to Mock
        return MockLLMProvider(model_name=settings.LLM_MODEL_NAME)

def get_llm_provider() -> BaseLLMProvider:
    return LLMFactory.get_provider()

llm_provider: BaseLLMProvider = LLMFactory.get_provider()
