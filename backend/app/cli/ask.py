import argparse
import asyncio
from app.core.config import settings
from app.core.logging import setup_logging
from app.llm.factory import LLMFactory
from app.services.ask_service import AskService

async def main_async(args: argparse.Namespace) -> None:
    setup_logging()

    llm_name = args.llm_provider or settings.LLM_PROVIDER
    llm_inst = LLMFactory.get_provider(provider_name=llm_name)

    ask_service = AskService(llm_provider_inst=llm_inst)
    response = await ask_service.ask(
        question=args.query,
        top_k=args.top_k,
        min_score=args.min_score
    )

    print("\nQuestion:")
    print(response.question)

    print("\nAnswer:")
    print(response.answer)

    print("\nEvidence:\n")
    if not response.evidence:
        print("No evidence chunks retrieved.")
    else:
        for ev in response.evidence:
            line_range = f"Lines {ev.start_line}-{ev.end_line}"
            print(f"{ev.filepath}")
            print(f"{line_range}\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoMind Ask CLI (Retrieval-Augmented Generation / RAG)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=False,
        help="Question or query string"
    )
    parser.add_argument(
        "positional_query",
        nargs="?",
        type=str,
        default=None,
        help="Positional query string"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of retrieved context chunks (default: 5)"
    )
    parser.add_argument(
        "--llm-provider", "-l",
        default=None,
        help="LLM provider choice (default: mock)"
    )
    parser.add_argument(
        "--min-score", "-s",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold (default: 0.0)"
    )

    args = parser.parse_args()
    # Resolve query from either --query flag or positional argument
    args.query = args.query or args.positional_query

    if not args.query:
        parser.error("Please provide a question via --query or positional argument.")

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
