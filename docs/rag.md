# EchoMind RAG (Retrieval-Augmented Generation) Pipeline

EchoMind features an offline-first **Retrieval-Augmented Generation (RAG)** pipeline allowing developers and users to ask questions over their indexed digital memory and codebase.

---

## 🏛️ RAG Architecture Overview

```
User Question
     │
     ▼
┌───────────────────────────┐
│       SearchService       │ (Dense vector search via Qdrant / local JSONL)
└────────────┬──────────────┘
             │ Top-k Chunks (Filepath, Start/End Lines, Content)
             ▼
┌───────────────────────────┐
│       PromptBuilder       │ (Formats context blocks with file citations)
└────────────┬──────────────┘
             │ Structured Context Prompt
             ▼
┌───────────────────────────┐
│      BaseLLMProvider      │ (MockLLMProvider / OpenAI / Anthropic / OpenRouter)
└────────────┬──────────────┘
             │ Generated Answer & Evidence Citations
             ▼
      CLI / API Response
```

---

## ⚡ CLI Usage

Execute natural language question answering directly from your command line:

```bash
python -m app.cli.ask --query "Where is QwenEmbeddingProvider implemented?"
```

Or pass positional query parameters:

```bash
python -m app.cli.ask "How does EchoMind generate embeddings?"
```

### Example Output
```text
Question:
Where is QwenEmbeddingProvider implemented?

Answer:
Based on the retrieved EchoMind codebase context, 'Where is QwenEmbeddingProvider implemented?' is implemented in backend/app/embeddings/qwen_provider.py.
Primary implementation details can be found in 'backend/app/embeddings/qwen_provider.py' (Lines 1-75).

Evidence:

backend/app/embeddings/factory.py
Lines 10-35

backend/app/embeddings/qwen_provider.py
Lines 1-75
```

---

## ⚙️ Configuration & Zero-API Key Mock Mode

The pipeline defaults to `LLM_PROVIDER=mock`, requiring zero API keys, network requests, or model downloads:

```ini
LLM_PROVIDER=mock
LLM_MODEL_NAME=mock-gpt-4o
```

Future providers (`openai`, `anthropic`, `openrouter`) plug seamlessly into the `BaseLLMProvider` interface without breaking existing code.
