# Configuration Reference

EchoMind manages configuration settings via Pydantic BaseSettings in `backend/app/core/config.py`. Settings can be overridden using environment variables or a local `.env` file in `backend/`.

---

## Environment Variables

| Variable | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `PROJECT_NAME` | `str` | Name of the project application | `EchoMind` |
| `EMBEDDING_PROVIDER` | `str` | Active embedding provider (`mock`, `qwen`, `bge`, `openai`) | `mock` |
| `EMBEDDING_MODEL_NAME` | `str` | Hugging Face model repository path or OpenAI model name | `Qwen/Qwen3-Embedding-8B` |
| `EMBEDDING_DIMENSION` | `int` | Vector dimension override for local Mock provider | `384` |
| `LLM_PROVIDER` | `str` | Active LLM backend provider (`mock`, `openai`) | `mock` |
| `OPENAI_BASE_URL` | `str` | HTTP endpoint base URL for OpenAI-compatible gateway | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | `str` | API key authentication credential for OpenAI gateway | `None` |
| `OPENAI_MODEL_NAME` | `str` | Model identifier string sent to OpenAI Chat Completion API | `gpt-4o-mini` |
| `POSTGRES_SERVER` | `str` | PostgreSQL database host address | `localhost` |
| `POSTGRES_PORT` | `int` | PostgreSQL database port | `5432` |
| `POSTGRES_DB` | `str` | PostgreSQL database name | `echomind_db` |
| `POSTGRES_USER` | `str` | PostgreSQL username credential | `echomind_user` |
| `POSTGRES_PASSWORD` | `str` | PostgreSQL password credential | `echomind_password` |
| `QDRANT_HOST` | `str` | Qdrant vector store endpoint host | `localhost` |
| `QDRANT_PORT` | `int` | Qdrant REST API port | `6333` |
| `QDRANT_COLLECTION_NAME` | `str` | Qdrant vector collection identifier | `echomind_embeddings` |
| `NEO4J_URI` | `str` | Neo4j graph database Bolt protocol URI | `bolt://localhost:7687` |
| `NEO4J_USER` | `str` | Neo4j graph database username | `neo4j` |
| `NEO4J_PASSWORD` | `str` | Neo4j graph database password | `echomind_neo4j_password` |

---

## Provider Selection Matrix

### Embedding Providers
- `mock`: Default zero-download hash provider for rapid testing on macOS (384 dimensions).
- `qwen`: High-precision 4096-dimension embeddings using `Qwen/Qwen3-Embedding-8B` optimized for GPU environments (A100).
- `bge`: 1024-dimension embeddings using `BAAI/bge-m3`.
- `openai`: OpenAI `text-embedding-3-small` (1536 dimensions).

### LLM Providers
- `mock`: Pre-formatted template generator for local offline CLI testing.
- `openai`: Connects to OpenAI APIs or custom vLLM / Ollama OpenAI-compatible endpoints.

---

## References

- [Architecture Guide](architecture.md)
- [API & CLI Reference](api.md)
