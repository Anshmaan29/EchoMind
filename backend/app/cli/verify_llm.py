"""
EchoMind Real LLM Gateway Verification CLI — Phase 4.0

Verifies connectivity and runtime response from configured OpenAI-compatible endpoint.

Usage:
    python -m app.cli.verify_llm
"""
import argparse
import asyncio
import time

from app.core.config import settings
from app.core.exceptions import EchoMindException
from app.core.logging import logger, setup_logging
from app.llm.factory import LLMFactory


async def main_async(args: argparse.Namespace) -> None:
    setup_logging()

    print("\n" + "=" * 65)
    print("🤖 ECHOMIND PRODUCTION LLM GATEWAY VERIFICATION")
    print("=" * 65)
    print(f"  LLM Provider      : {settings.LLM_PROVIDER}")
    print(f"  Model             : {settings.LLM_MODEL}")
    print(f"  Base URL          : {settings.LLM_BASE_URL}")
    print(f"  Streaming Enabled : {settings.LLM_STREAM}")
    print(f"  Temperature       : {settings.LLM_TEMPERATURE}")
    print(f"  Max Tokens        : {settings.LLM_MAX_TOKENS}")
    print("=" * 65 + "\n")

    provider = LLMFactory.get_provider()
    test_query = "Reply only with the word READY."

    print(f"Sending verification probe to endpoint: '{settings.LLM_BASE_URL}'...")
    start_time = time.perf_counter()

    try:
        response = await provider.generate_answer(query=test_query, context_prompt=test_query)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        word_count = len(response.split())
        char_count = len(response)

        print("\n✅ Endpoint verification successful!\n")
        print(f"  Latency        : {latency_ms:.2f} ms")
        print(f"  Response Length: {char_count} chars (~{word_count} words)")
        print(f"  Response Output:\n  >>> {response}\n")

    except EchoMindException as exc:
        print(f"\n❌ Verification failed with EchoMindException: {exc.message}")
        if exc.details:
            print(f"   Details: {exc.details}")
        raise
    except Exception as exc:
        print(f"\n❌ Verification failed with unexpected error: {exc}")
        raise EchoMindException(f"LLM Endpoint Verification Failed: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoMind Production LLM Gateway Verification CLI"
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
