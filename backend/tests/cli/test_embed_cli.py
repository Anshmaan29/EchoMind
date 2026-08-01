import os
import argparse
import pytest
from app.cli.embed import scan_input_path, main_async, chunk_file_lines

def test_chunk_file_lines_metadata(tmp_path) -> None:
    sample_file = tmp_path / "Component.tsx"
    sample_file.write_text(
        "import React from 'react';\n"
        "export const Component = () => {\n"
        "  return <div>EchoMind</div>;\n"
        "};\n",
        encoding="utf-8"
    )

    chunks = chunk_file_lines(str(sample_file))
    assert len(chunks) >= 1
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] >= 1
    assert "EchoMind" in chunks[0]["content"]

@pytest.mark.asyncio
async def test_scan_input_path_recursive_extensions(tmp_path) -> None:
    # Create nested folder structure with multiple source file types
    backend_dir = tmp_path / "backend" / "app"
    frontend_dir = tmp_path / "frontend" / "components"
    backend_dir.mkdir(parents=True)
    frontend_dir.mkdir(parents=True)

    py_file = backend_dir / "service.py"
    py_file.write_text("def process_memory():\n    return 'memory_ok'\n", encoding="utf-8")

    tsx_file = frontend_dir / "Header.tsx"
    tsx_file.write_text("export const Header = () => <h1>EchoMind Header</h1>;\n", encoding="utf-8")

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("version: '1.0'\nname: echomind\n", encoding="utf-8")

    md_file = tmp_path / "README.md"
    md_file.write_text("# EchoMind Memory System\n\nRecursive embedding scan test.", encoding="utf-8")

    items = scan_input_path(str(tmp_path))
    assert len(items) >= 4

    extensions = {it.meta_data.get("extension") for it in items}
    assert ".py" in extensions
    assert ".tsx" in extensions
    assert ".yaml" in extensions
    assert ".md" in extensions

    for item in items:
        assert "start_line" in item.meta_data
        assert "end_line" in item.meta_data
        assert "filepath" in item.meta_data

@pytest.mark.asyncio
async def test_embed_cli_main_execution(tmp_path) -> None:
    sample_code = tmp_path / "main.py"
    sample_code.write_text("print('Executing EchoMind Codebase Embedding CLI')", encoding="utf-8")

    args = argparse.Namespace(
        input=str(sample_code),
        provider=None,
        batch_size=16,
        workers=2,
        collection_name="test_codebase_collection",
        no_resume=True
    )

    await main_async(args)
