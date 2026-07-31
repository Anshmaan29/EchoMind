# EchoMind Backend - Hybrid Vector Search & AI Operating System

EchoMind is an AI-powered Digital Memory Operating System. Instead of simply chatting over static documents, it reconstructs a user's digital history (PDFs, GitHub repositories, WhatsApp exports, Audio recordings, Notes) into an intelligent, queryable **Temporal Memory Graph**.

---

## 🏛️ Architecture Overview

```
                  ┌──────────────────────────────────────────────┐
                  │                 FastAPI API                  │
                  │   GET /health  | POST /upload/pdf            │
                  │   GET /entities| GET /relationships           │
                  │   GET /timeline| GET /graph/search           │
                  └──────────────────────┬───────────────────────┘
                                         │ (Pydantic v2 DTOs & DI)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │              Services Layer                  │
                  │ IngestionService | DocumentService           │
                  │ KnowledgeService | TimelineService           │
                  │ SearchService (Hybrid Vector & JSONL Backup) │
                  └──────────────────────┬───────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────┐         ┌──────────────────────────────┐         ┌──────────────┐
│  Ingestion   │         │   Temporal Knowledge Engine  │         │  Reasoning   │
│  Pipeline    │         ├──────────────────────────────┤         │    Layer     │
│ Load->Parse  │         │  Entity Extractor (23 Types) │         │ Entity/Time  │
│ ->Chunk      │         │  Rel Extractor (16 Types)    │         │ Rel/Project/ │
└───────┬──────┘         │  Timeline Engine & Replay    │         │ Memory Fusion│
        │                └──────────────┬───────────────┘         └──────┬───────┘
        │                               │                                │
        ▼                               ▼                                ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                   CLI Tools & AI Kosh GPU Experiments Harness                   │
│   app.cli.embed (Recursive Scan)  |  app.cli.search (Vector & Backup Search)   │
└───────┬───────────────────────────────┬────────────────────────────────┬───────┘
        │                               │                                │
        ▼                               ▼                                ▼
┌────────────────┐            ┌──────────────────┐            ┌──────────────────┐
│ Vector Engine  │            │  Knowledge Graph │            │   PostgreSQL DB  │
│ Qdrant Store   │            │ Neo4j Cypher     │            │ Document, Chunk, │
│ Cosine Search  │            │ Subgraph Traversal│            │ Entity, Rel, Time│
└────────────────┘            └──────────────────┘            └──────────────────┘
```

---

## 🔍 Hybrid Vector Search Service & CLI

The **Hybrid Search Service** (`app/services/search_service.py`) provides provider-agnostic cosine similarity search over code and document chunks. It queries Qdrant when online, and automatically falls back to local JSONL vector backup files (`embeddings_backup.jsonl`).

### CLI Search Usage

Search your indexed repository directly from the command line:

```bash
python -m app.cli.search --query "How does the embedding pipeline work?" --top-k 5
```

CLI Parameters:
- `--query` / `-q`: Natural language search query string (required)
- `--top-k` / `-k`: Number of top matching chunks to return (default: `5`)
- `--backup-path` / `-f`: Path to custom JSONL backup file
- `--min-score` / `-s`: Minimum cosine similarity score threshold (default: `0.0`)

---

## 🚀 Recursive Codebase Ingestion CLI

Recursively scan and chunk all supported source files (`.py`, `.md`, `.json`, `.yaml`, `.yml`, `.toml`, `.tsx`, `.ts`, `.js`, `.jsx`):

```bash
python -m app.cli.embed --input ..
```
