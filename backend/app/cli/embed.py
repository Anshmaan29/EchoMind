import argparse
import asyncio
import os
import sys
import uuid
from typing import Any
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.embeddings.factory import EmbeddingFactory
from app.embeddings.pipeline import EmbeddingItem, GenericEmbeddingPipeline, SourceType

SUPPORTED_EXTENSIONS = {
    ".py": "github",
    ".tsx": "github",
    ".ts": "github",
    ".jsx": "github",
    ".js": "github",
    ".md": "pdf",
    ".json": "pdf",
    ".yaml": "pdf",
    ".yml": "pdf",
    ".toml": "pdf",
}

IGNORED_DIRS = {
    ".git",
    "node_modules",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    ".checkpoints",
    "outputs",
    "logs",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    "data",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}

def extract_code_symbols(filepath: str, content: str) -> dict[str, Any]:
    """Extracts class names, function names, and imported symbols from Python source files."""
    import re
    symbols: dict[str, Any] = {
        "class_names": [],
        "function_names": [],
        "imported_symbols": [],
        "defined_symbols": [],
    }
    if not filepath.endswith(".py"):
        return symbols

    # Class definitions
    symbols["class_names"] = re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", content, re.MULTILINE)
    # Function definitions
    symbols["function_names"] = re.findall(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", content, re.MULTILINE)
    # Import statements
    imports = re.findall(r"^(?:from\s+\S+\s+import|import)\s+(.+)", content, re.MULTILINE)
    for imp in imports:
        for sym in re.split(r"[,\s]+", imp):
            sym = sym.strip().split(" as ")[0].strip()
            if sym:
                symbols["imported_symbols"].append(sym)
    # All class + function names together
    symbols["defined_symbols"] = symbols["class_names"] + symbols["function_names"]
    return symbols


def chunk_file_lines(
    filepath: str,
    max_lines_per_chunk: int = 40,
    max_chars_per_chunk: int = 1500
) -> list[dict[str, Any]]:
    """Reads a file line by line and produces chunk windows with exact start_line and end_line metadata."""
    chunks: list[dict[str, Any]] = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning(f"Could not read file '{filepath}': {e}")
        return chunks

    if not lines:
        return chunks

    current_lines: list[str] = []
    current_char_count = 0
    start_line = 1

    for line_idx, line in enumerate(lines, start=1):
        if (
            len(current_lines) >= max_lines_per_chunk
            or (current_char_count + len(line) > max_chars_per_chunk and current_lines)
            or (line.strip() == "" and len(current_lines) >= 20)
        ):
            chunk_text = "".join(current_lines).strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "start_line": start_line,
                    "end_line": line_idx - 1
                })
            current_lines = [line]
            current_char_count = len(line)
            start_line = line_idx
        else:
            current_lines.append(line)
            current_char_count += len(line)

    if current_lines:
        chunk_text = "".join(current_lines).strip()
        if chunk_text:
            chunks.append({
                "content": chunk_text,
                "start_line": start_line,
                "end_line": len(lines)
            })

    return chunks

def scan_input_path(input_path: str) -> list[EmbeddingItem]:
    """Recursively scans input directory or file for supported source files and generates line-annotated EmbeddingItems."""
    items: list[EmbeddingItem] = []
    abs_input_path = os.path.abspath(input_path)

    if not os.path.exists(abs_input_path):
        logger.warning(f"Input path '{input_path}' (resolved: '{abs_input_path}') does not exist.")
        return items

    files_to_process: list[str] = []
    if os.path.isfile(abs_input_path):
        files_to_process.append(abs_input_path)
    else:
        for root, dirs, files in os.walk(abs_input_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS or file in ["README.md", "docker-compose.yml", "Dockerfile", ".env.example"]:
                    files_to_process.append(os.path.join(root, file))

    for filepath in files_to_process:
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        source_type: SourceType = SUPPORTED_EXTENSIONS.get(ext, "github" if ext in [".py", ".ts", ".tsx", ".js", ".jsx"] else "pdf")

        # Read full file content for symbol extraction
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as _f:
                full_content = _f.read()
        except Exception:
            full_content = ""

        file_symbols = extract_code_symbols(filepath, full_content)

        chunk_blocks = chunk_file_lines(filepath)
        for c in chunk_blocks:
            item_id = f"{filename}_L{c['start_line']}_L{c['end_line']}_{uuid.uuid4().hex[:6]}"
            # Extract chunk-level symbols too
            chunk_symbols = extract_code_symbols(filepath, c["content"])
            items.append(
                EmbeddingItem(
                    id=item_id,
                    source=source_type,
                    content=c["content"],
                    meta_data={
                        "filepath": filepath,
                        "filename": filename,
                        "extension": ext or "none",
                        "start_line": c["start_line"],
                        "end_line": c["end_line"],
                        "class_names": chunk_symbols["class_names"],
                        "function_names": chunk_symbols["function_names"],
                        "file_class_names": file_symbols["class_names"],
                        "file_function_names": file_symbols["function_names"],
                        "defined_symbols": chunk_symbols["defined_symbols"],
                        "imported_symbols": chunk_symbols["imported_symbols"],
                    }
                )
            )

    logger.info(f"Scanned path '{input_path}' ({len(files_to_process)} files) -> Created {len(items)} line-annotated EmbeddingItems.")
    return items

async def main_async(args: argparse.Namespace) -> None:
    setup_logging()
    logger.info(f"Starting EchoMind Codebase Embedding CLI on path '{args.input}'...")

    provider_name = args.provider or settings.EMBEDDING_PROVIDER
    provider = EmbeddingFactory.get_provider(provider_name=provider_name)

    items = scan_input_path(args.input)
    if not items:
        logger.warning(f"No valid source files found under '{args.input}'. Exiting CLI.")
        return

    pipeline = GenericEmbeddingPipeline(embedder=provider)
    metrics = await pipeline.process_items(
        items=items,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        max_workers=args.workers,
        resume=not args.no_resume
    )

    device_info = getattr(provider, "device", "cpu").upper()
    model_name = getattr(provider, "model_name", "hash_mock")

    print("\n" + "=" * 65)
    print("🚀 ECHOMIND EMBEDDING CLI SUMMARY")
    print("=" * 65)
    print(f"Embedding Provider   : {provider_name.upper()}")
    print(f"Embedding Model      : {model_name}")
    print(f"Device               : {device_info}")
    print(f"Embedding Dimension  : {provider.dimension}")
    print(f"Chunks Processed     : {metrics.processed_items}")
    print(f"Embeddings/sec       : {metrics.embeddings_per_sec}")
    print("=" * 65 + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoMind Recursive Codebase Embedding CLI"
    )
    parser.add_argument(
        "--input", "-i",
        default="..",
        help="Path to input file or directory to scan recursively (default: ..)"
    )
    parser.add_argument(
        "--provider", "-p",
        default=None,
        help="Embedding provider override (mock, qwen, openai, bge)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size for acceleration (default: 64)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of async worker tasks (default: 4)"
    )
    parser.add_argument(
        "--collection-name", "-c",
        default=settings.QDRANT_COLLECTION_NAME,
        help="Target Qdrant collection name"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resumption and reprocess all items"
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
