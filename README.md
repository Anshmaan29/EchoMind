# EchoMind - AI-powered Digital Memory Operating System

EchoMind is an AI-powered Digital Memory Operating System. Instead of simply chatting over static documents, it reconstructs a user's digital history (PDFs, GitHub repositories, WhatsApp exports, Audio recordings, Notes) into an intelligent, queryable **Temporal Memory Graph**.

## 🚀 Milestones Completed

- **Milestone 1**: Clean Architecture Backend (FastAPI + PostgreSQL 16 + Qdrant + Docker Compose)
- **Milestone 2**: Temporal Knowledge Engine (Entity & Relationship Extractors + Timeline Engine + Neo4j GraphStore)
- **Milestone 3**: Production Embedding Generation (BAAI/bge-m3 + Qwen/Qwen3-Embedding-8B + Local JSONL Backups)
- **AI Kosh GPU Experiments Harness (`experiments/`)**: Standalone GPU execution runners for embedding generation, entity extraction, relationship discovery, vector re-ranking, and RAG evaluation.
- **EchoMind Ask CLI (RAG Pipeline)**: Conversational RAG interface connecting `SearchService`, `PromptBuilder`, and `LLMProvider`.

### Ask CLI Usage
```bash
python -m app.cli.ask --query "Where is QwenEmbeddingProvider implemented?"
```

For full architecture diagrams, see the [Backend Documentation](file:///Users/anshmaansingh/Echomind/backend/README.md) and [RAG Guide](file:///Users/anshmaansingh/Echomind/docs/rag.md).
