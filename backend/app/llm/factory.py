from app.core.config import settings
from app.core.exceptions import EchoMindException
from app.llm.providers.base import BaseLLMProvider
from app.llm.providers.huggingface import HuggingFaceLLMProvider
from app.llm.providers.mock import MockLLMProvider
from app.llm.providers.openai_compatible import OpenAICompatibleProvider


class LLMFactory:
    """
    Factory providing LLM Provider instances based on configuration.
    """

    @staticmethod
    def get_provider(provider_name: str | None = None) -> BaseLLMProvider:
        provider_type = (provider_name or settings.LLM_PROVIDER).lower().strip()

        if provider_type == "mock":
            return MockLLMProvider(model_name=settings.LLM_MODEL)

        if provider_type in ("huggingface", "hf"):
            return HuggingFaceLLMProvider(
                model_name=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )

        if provider_type in ("openai", "openai_compatible", "openai-compatible"):
            return OpenAICompatibleProvider(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                stream=settings.LLM_STREAM,
            )

        raise EchoMindException(
            f"Unsupported LLM Provider '{provider_name or settings.LLM_PROVIDER}'. "
            f"Supported providers: 'mock', 'huggingface', 'openai'."
        )


def get_llm_provider() -> BaseLLMProvider:
    return LLMFactory.get_provider()


llm_provider: BaseLLMProvider = LLMFactory.get_provider()
