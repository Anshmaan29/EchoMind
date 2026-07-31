import argparse
import pytest
from app.llm.mock_provider import MockLLMProvider
from app.rag.prompt_builder import PromptBuilder
from app.services.ask_service import AskService
from app.services.search_service import SearchResult
from app.cli.ask import main_async

def test_prompt_builder_formatting() -> None:
    results = [
        SearchResult(
            id="c1",
            filepath="backend/app/embeddings/factory.py",
            filename="factory.py",
            start_line=10,
            end_line=35,
            score=0.92,
            content="def get_provider(): pass",
            source="github"
        )
    ]

    prompt = PromptBuilder.build_context_prompt("How does factory work?", results)
    assert "User Question: How does factory work?" in prompt
    assert "Filepath: backend/app/embeddings/factory.py" in prompt
    assert "Line Range: L10-L35" in prompt
    assert "def get_provider(): pass" in prompt

@pytest.mark.asyncio
async def test_mock_llm_provider() -> None:
    provider = MockLLMProvider()
    results = [
        SearchResult(
            id="c1",
            filepath="backend/app/embeddings/qwen_provider.py",
            filename="qwen_provider.py",
            start_line=40,
            end_line=90,
            score=0.95,
            content="class QwenEmbeddingProvider: pass",
            source="github"
        )
    ]

    answer = await provider.generate_answer(
        query="Where is Qwen provider?",
        context_prompt="Context prompt",
        results=results
    )

    assert "backend/app/embeddings/qwen_provider.py" in answer
    assert "Lines 40-90" in answer

@pytest.mark.asyncio
async def test_ask_service_pipeline(tmp_path) -> None:
    ask_service = AskService()
    response = await ask_service.ask(question="EchoMind architecture", top_k=2)

    assert response.question == "EchoMind architecture"
    assert isinstance(response.answer, str)
    assert isinstance(response.evidence, list)

@pytest.mark.asyncio
async def test_ask_cli_main_execution() -> None:
    args = argparse.Namespace(
        query="Where is QwenEmbeddingProvider implemented?",
        positional_query=None,
        top_k=3,
        llm_provider="mock",
        min_score=0.0
    )

    await main_async(args)
