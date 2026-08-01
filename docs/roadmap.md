# Project Roadmap

This document outlines completed milestones, work in progress, and planned developments for EchoMind.

---

## Phase 1: Core Storage & Retrieval Architecture
- [x] **FastAPI Application Core**: Async REST API endpoints, database initialization, and settings configuration.
- [x] **Vector Store Abstraction**: Qdrant vector database integration with automatic local JSONL backup fallback.
- [x] **Hybrid Search Engine**: Combined dense vector cosine similarity (0.55 weight), BM25 keyword matching, and structural path re-ranking.
- [x] **Dependency Injection**: Modular interfaces (`BaseEmbeddingProvider`, `BaseVectorStore`, `BaseLLMProvider`).

## Phase 2: Temporal & Code Intelligence
- [x] **Git Connector**: Automated commit history, author, diff stats, and modified file indexing.
- [x] **Timeline Service**: Chronological memory aggregator for project commits, notes updates, and document modifications.
- [x] **Timeline-Aware Ask CLI**: Integrated CLI routing query temporal intent directly to `TimelineService`.

## Phase 3: Knowledge Base & Production LLM Gateway
- [x] **Personal Notes Connector**: Recursive scanner for Markdown (`.md`) and text (`.txt`) notes with title, heading, tag, and link extraction.
- [x] **Dynamic Embedding Dimensions**: Provider-native default dimensions (Qwen: 4096, OpenAI: 1536, BGE: 1024, Mock: 384) without global overrides.
- [x] **OpenAI-Compatible LLM Gateway**: Provider implementation supporting OpenAI Chat Completion endpoints and response streaming.
- [x] **GPU Inference Stability**: Memory lifecycle optimization using `torch.inference_mode()`, explicit tensor deletion, and empty_cache cleanup.

## Phase 4: Multi-Source Expansion & Dashboard (Planned)
- [ ] **WhatsApp Chat Export Connector**: Ingestion module for exported chat transcripts.
- [ ] **Audio Transcription Pipeline**: Automated speech-to-text processing for voice notes.
- [ ] **Web Application Frontend**: Responsive interactive UI dashboard for timeline inspection and search.
- [ ] **Graph Visualization Engine**: Interactive visual explorer for Neo4j entity-relationship networks.

---

## References

- [Architecture Guide](architecture.md)
- [Development Principles](development.md)
