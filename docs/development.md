# Development Principles & Contributing Guide

This document outlines core software engineering practices and instructions for contributing to EchoMind.

---

## Core Engineering Principles

### 1. Modular Architecture & Provider Abstraction
All core subsystems (vector stores, embedding engines, LLM backends, graph stores) implement explicit abstract base interfaces:
- `BaseEmbeddingProvider` (`app.embeddings.base`)
- `BaseVectorStore` (`app.vector.base`)
- `BaseLLMProvider` (`app.llm.providers.base`)
- `BaseGraphStore` (`app.graph.base`)

Downstream search, timeline, and RAG services interact exclusively through these abstractions, enabling zero-code changes when swapping underlying providers.

### 2. Dependency Injection & Lazy Initialization
Services receive instantiated dependencies rather than creating global hardcoded instances. Embedding providers use lazy factory initialization (`EmbeddingFactory.get_provider()`) with process-level singleton caching, ensuring heavy machine learning models (like Qwen) are imported and initialized on-demand and loaded only once per process.

### 3. Grounded Retrieval
All answer generation workflows prioritize verifiable context extracted from personal data sources (git commits, markdown notes, PDFs). Answers cite explicit file paths, line numbers, and commit hashes.

### 4. Production-First Design & Fallback Resiliency
Systems include graceful offline fallbacks:
- If Qdrant vector database is unavailable, operations transparently fallback to local JSONL vector storage (`data/embeddings_backup.jsonl`).
- If Neo4j graph store is unreachable, operations fallback to in-memory mock graph stores.
- GPU inference uses `torch.inference_mode()` with post-batch CUDA memory synchronization and cleanup to prevent allocator fragmentation on memory-constrained GPU environments.

---

## Contributing Guide

Contributions are welcome. Please follow these guidelines:

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/Anshmaan29/EchoMind.git
   cd EchoMind
   ```

2. **Branching Model**:
   Create a focused feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Code Style & Guidelines**:
   - Write clean, type-annotated Python 3.11 code.
   - Maintain docstrings and maintain docstrings across edits.
   - Never introduce hardcoded model dimensions or global static paths.

4. **Testing**:
   Run the pytest test suite from `backend/`:
   ```bash
   cd backend
   pytest
   ```

5. **Pull Request Submission**:
   Submit a PR summarizing:
   - What problem your change solves
   - Implementation details and design rationale
   - Manual or automated verification steps performed

---

## References

- [Architecture Guide](architecture.md)
- [Roadmap](roadmap.md)
