# EchoMind - AI-powered Digital Memory Operating System

EchoMind is an AI-powered Digital Memory Operating System. Instead of simply chatting over static documents, it reconstructs a user's digital history (PDFs, GitHub repositories, WhatsApp exports, Audio recordings, Notes) into an intelligent, queryable **Temporal Memory Graph**.

## 🚀 Milestones Completed

- **Milestone 1**: Clean Architecture Backend (FastAPI + PostgreSQL 16 + Qdrant + Docker Compose)
- **Milestone 2**: Temporal Knowledge Engine (Entity & Relationship Extractors + Timeline Engine + Neo4j GraphStore)
- **Milestone 3.1 & 3.2**: Production-Grade Embedding Pipeline (BAAI/bge-m3 + PyTorch CUDA acceleration + Checkpoint Ledger + Local JSONL Backups)
- **AI Kosh GPU Experiments Framework (`experiments/`)**: Standalone GPU execution runners for embedding generation, entity extraction, relationship discovery, vector re-ranking, and RAG evaluation.

### Running Standalone GPU Experiments
```bash
PYTHONPATH=backend:. uv run python experiments/embedding_generation/run_embedding_experiment.py
```

### Running Backend API & Services
```bash
docker-compose up -d --build
curl http://localhost:8000/health
```

For full architecture diagrams and experiment guides, see the [Backend Documentation](file:///Users/anshmaansingh/Echomind/backend/README.md).
