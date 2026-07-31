import argparse
import asyncio
from app.core.logging import setup_logging
from app.services.search_service import SearchService

async def main_async(args: argparse.Namespace) -> None:
    setup_logging()

    search_service = SearchService(backup_filepath=args.backup_path)
    results = await search_service.search(
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score
    )

    print("\n" + "=" * 70)
    print(f"🔍 ECHOMIND VECTOR SEARCH RESULTS FOR: \"{args.query}\"")
    print("=" * 70)

    if not results:
        print("No matching code or document chunks found.")
        print("Tip: Run 'python -m app.cli.embed --input ..' to index your codebase first.")
        print("=" * 70 + "\n")
        return

    for idx, res in enumerate(results, start=1):
        line_info = f"L{res.start_line}-L{res.end_line}"
        print(f"[{idx}] Score: {res.score:.4f} | {res.filepath}:{line_info}")
        print(f"    Source: {res.source} | File: {res.filename}")
        print("    " + "-" * 62)
        
        # Format snippet lines
        content_lines = res.content.splitlines()
        preview_lines = content_lines[:6]
        for line in preview_lines:
            print(f"    │ {line}")
        if len(content_lines) > 6:
            print(f"    │ ... ({len(content_lines) - 6} more lines)")
        print("=" * 70)

    print(f"\nReturned {len(results)} top matching chunks.\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoMind Codebase & Memory Vector Search CLI"
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Natural language query string or code search term"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of top search results to return (default: 5)"
    )
    parser.add_argument(
        "--backup-path", "-f",
        default=None,
        help="Path to JSONL embeddings backup file"
    )
    parser.add_argument(
        "--min-score", "-s",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold (default: 0.0)"
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
