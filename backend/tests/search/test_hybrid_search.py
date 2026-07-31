"""
Regression tests for hybrid search ranking correctness.
These tests ensure:
  - qwen_provider.py ranks #1 for "QwenEmbeddingProvider" query
  - noise paths (.pytest_cache, tests/, __pycache__) are never ranked above real source files
  - duplicate (filepath, start_line, end_line) results are eliminated
"""
import os
import json
import asyncio
import tempfile
import pytest
from app.services.search_service import SearchService, SearchResult, _compute_hybrid_boost, _tokenize_query, _is_noise_path
from app.embeddings.mock_provider import MockEmbeddingProvider


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

def test_tokenize_query_splits_camelcase():
    tokens = _tokenize_query("QwenEmbeddingProvider")
    assert "qwen" in tokens
    assert "embedding" in tokens
    assert "provider" in tokens


def test_tokenize_query_lowercased():
    tokens = _tokenize_query("SearchService")
    assert all(t == t.lower() for t in tokens)


def test_is_noise_path_pytest_cache():
    assert _is_noise_path("/Users/foo/.pytest_cache/README.md")
    assert _is_noise_path("/project/.pytest_cache/v/cache/lastfailed")


def test_is_noise_path_pycache():
    assert _is_noise_path("/project/__pycache__/foo.pyc")


def test_is_noise_path_tests_dir():
    assert _is_noise_path("/project/backend/tests/test_foo.py")


def test_is_noise_path_real_source():
    assert not _is_noise_path("/project/backend/app/embeddings/qwen_provider.py")
    assert not _is_noise_path("/project/backend/app/services/search_service.py")


def test_hybrid_boost_filename_match():
    tokens = _tokenize_query("QwenEmbeddingProvider")
    meta = {"filepath": "/backend/app/embeddings/qwen_provider.py", "filename": "qwen_provider.py", "class_names": [], "function_names": [], "file_class_names": [], "defined_symbols": [], "imported_symbols": []}
    rec = {"content": "class QwenEmbeddingProvider:"}
    boost = _compute_hybrid_boost(tokens, "QwenEmbeddingProvider", rec, meta)
    # Expect a positive boost from filename + content match
    assert boost > 0.25


def test_hybrid_boost_noise_path_penalized():
    tokens = _tokenize_query("QwenEmbeddingProvider")
    meta = {"filepath": "/backend/.pytest_cache/README.md", "filename": "README.md", "class_names": [], "function_names": [], "file_class_names": [], "defined_symbols": [], "imported_symbols": []}
    rec = {"content": "QwenEmbeddingProvider"}
    boost = _compute_hybrid_boost(tokens, "QwenEmbeddingProvider", rec, meta)
    # Even with exact match in content, noise penalty should drag it below 0
    assert boost < 0.0


def test_hybrid_boost_class_name_match():
    tokens = _tokenize_query("QwenEmbeddingProvider")
    meta = {"filepath": "/backend/app/embeddings/qwen_provider.py", "filename": "qwen_provider.py",
            "class_names": ["QwenEmbeddingProvider"], "function_names": [], "file_class_names": ["QwenEmbeddingProvider"], "defined_symbols": ["QwenEmbeddingProvider"], "imported_symbols": []}
    rec = {"content": "class QwenEmbeddingProvider(BaseEmbeddingProvider):"}
    boost = _compute_hybrid_boost(tokens, "QwenEmbeddingProvider", rec, meta)
    assert boost >= 0.40  # filename + class + content match


# ---------------------------------------------------------------------------
# Integration test: search over a minimal JSONL fixture
# ---------------------------------------------------------------------------

QWEN_CHUNK = {
    "id": "qwen_provider_L1_L30_abc123",
    "source": "github",
    "content": "class QwenEmbeddingProvider(BaseEmbeddingProvider):\n    \"\"\"Qwen3-Embedding-8B provider.\"\"\"\n",
    "embedding_model": "hash_mock",
    "embedding_vector": None,  # filled in fixture
    "metadata": {
        "filepath": "/Users/anshmaansingh/Echomind/backend/app/embeddings/qwen_provider.py",
        "filename": "qwen_provider.py",
        "extension": ".py",
        "start_line": 1,
        "end_line": 30,
        "class_names": ["QwenEmbeddingProvider"],
        "function_names": [],
        "file_class_names": ["QwenEmbeddingProvider"],
        "defined_symbols": ["QwenEmbeddingProvider"],
        "imported_symbols": [],
    }
}

NOISE_CHUNK = {
    "id": "pytest_cache_L1_L8_xyz999",
    "source": "pdf",
    "content": "# pytest cache directory\nDo not commit to version control.\n",
    "embedding_model": "hash_mock",
    "embedding_vector": None,
    "metadata": {
        "filepath": "/Users/anshmaansingh/Echomind/backend/.pytest_cache/README.md",
        "filename": "README.md",
        "extension": ".md",
        "start_line": 1,
        "end_line": 8,
        "class_names": [],
        "function_names": [],
        "file_class_names": [],
        "defined_symbols": [],
        "imported_symbols": [],
    }
}

UNRELATED_CHUNK = {
    "id": "logging_L45_L46_def456",
    "source": "github",
    "content": 'logger: BoundLogger = structlog.get_logger("echomind")',
    "embedding_model": "hash_mock",
    "embedding_vector": None,
    "metadata": {
        "filepath": "/Users/anshmaansingh/Echomind/backend/app/core/logging.py",
        "filename": "logging.py",
        "extension": ".py",
        "start_line": 45,
        "end_line": 46,
        "class_names": [],
        "function_names": [],
        "file_class_names": [],
        "defined_symbols": [],
        "imported_symbols": [],
    }
}


@pytest.fixture
async def hybrid_search_service(tmp_path):
    """Build a SearchService backed by a minimal JSONL fixture with real embeddings."""
    provider = MockEmbeddingProvider(dimension=384)

    # Generate real embeddings for each chunk
    texts = [QWEN_CHUNK["content"], NOISE_CHUNK["content"], UNRELATED_CHUNK["content"]]
    vecs = await provider.embed_texts(texts)

    chunks = [QWEN_CHUNK, NOISE_CHUNK, UNRELATED_CHUNK]
    jsonl_path = str(tmp_path / "test_backup.jsonl")
    with open(jsonl_path, "w") as f:
        for chunk, vec in zip(chunks, vecs):
            chunk = dict(chunk)
            chunk["embedding_vector"] = vec
            f.write(json.dumps(chunk) + "\n")

    # Disable Qdrant by mocking is_available = False
    class _NullStore:
        is_available = False
        async def initialize_collection(self, **_): pass

    svc = SearchService(
        embedder=provider,
        vector_store_inst=_NullStore(),
        backup_filepath=jsonl_path
    )
    return svc


@pytest.mark.asyncio
async def test_search_qwen_provider_ranks_first(hybrid_search_service):
    """The qwen_provider.py chunk must rank #1 when searching 'QwenEmbeddingProvider'."""
    results = await hybrid_search_service.search("QwenEmbeddingProvider", top_k=3)
    assert len(results) > 0, "No results returned"
    assert "qwen_provider.py" in results[0].filepath, (
        f"Expected qwen_provider.py at rank #1, got: {results[0].filepath}"
    )


@pytest.mark.asyncio
async def test_noise_path_never_top(hybrid_search_service):
    """pytest_cache paths must never rank above real source files."""
    results = await hybrid_search_service.search("QwenEmbeddingProvider", top_k=3)
    noise = [r for r in results if ".pytest_cache" in r.filepath]
    real = [r for r in results if "qwen_provider.py" in r.filepath]
    if noise and real:
        assert real[0].score > noise[0].score, (
            f"Noise path {noise[0].filepath} (score={noise[0].score}) outranked "
            f"real source {real[0].filepath} (score={real[0].score})"
        )


@pytest.mark.asyncio
async def test_no_duplicate_results(hybrid_search_service):
    """No two results should have the same (filepath, start_line, end_line)."""
    results = await hybrid_search_service.search("QwenEmbeddingProvider", top_k=5)
    keys = [(r.filepath, r.start_line, r.end_line) for r in results]
    assert len(keys) == len(set(keys)), f"Duplicate results found: {keys}"
