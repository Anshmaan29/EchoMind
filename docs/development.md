# EchoMind macOS Local Development Guide

Welcome to local development on **EchoMind**. This guide explains how to set up your environment, run local embeddings without downloading heavy neural network models, execute test suites, and perform vector searches.

---

## 🏛️ Core Philosophy: Zero-Download & Zero-Qdrant Development

Local development on macOS is designed to be **fast, lightweight, clean, and offline-first**:
- **Mock Embedding Provider** (`EMBEDDING_PROVIDER=mock`): Zero LLM or transformer model weights are downloaded to your Mac. All embedding generation, CLI utilities, and tests use deterministic in-memory vector hashing.
- **Optional Qdrant Vector Store**: You do **not** need Qdrant or Docker running locally. When Qdrant is offline, EchoMind automatically logs a single warning (`"Qdrant unavailable. Using local JSONL backup."`) and streams vectors directly into `data/embeddings_backup.jsonl` and SQLite checkpoints without error noise or stack traces.

---

## ⚙️ Environment Configuration

Ensure your local `.env` and `backend/.env` files set the mock provider:

```ini
EMBEDDING_PROVIDER=mock
EMBEDDING_MODEL_NAME=mock_hash
EMBEDDING_DIMENSION=384
```

---

## 🚀 Running Local Embedding Generation

Recursively scan and chunk source files across your local repository without Qdrant or GPU:

```bash
cd backend
python -m app.cli.embed --input ..
```

### CLI Output (Clean Console)
```text
2026-08-01T04:45:00.000000Z [warning] Qdrant unavailable. Using local JSONL backup.

=================================================================
🚀 ECHOMIND EMBEDDING CLI SUMMARY
=================================================================
Embedding Provider   : MOCK
Embedding Model      : mock_hash
Device               : CPU
Embedding Dimension  : 384
Chunks Processed     : 302
Embeddings/sec       : 4314.29
=================================================================
```

---

## 🔍 Running Local JSONL Vector Search

Search your local codebase vectors stored in `data/embeddings_backup.jsonl` with zero Qdrant dependency:

```bash
python -m app.cli.search --query "Where is SearchService implemented?"
```

---

## 🧪 Running Pytest Test Suite

Execute the fast local test suite:

```bash
PYTHONPATH=backend uv run pytest backend/tests -v
```

---

## 🌿 Git & Contribution Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Verify all tests pass locally:
   ```bash
   PYTHONPATH=backend uv run pytest backend/tests -v
   ```
3. Commit and push:
   ```bash
   git add .
   git commit -m "feat: add feature"
   git push origin feature/my-feature
   ```
