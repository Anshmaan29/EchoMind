# EchoMind LLM Provider Architecture

EchoMind features a pluggable, environment-driven LLM Provider system (`BaseLLMProvider`).
It allows local development on macOS to remain zero-dependency and lightweight using a Mock Provider, while providing seamless production deployment paths for local open-weights HuggingFace models, OpenAI-compatible HTTP endpoints (vLLM, Ollama), and AI Kosh deployments.

---

## Provider Overview

```
                          ┌───────────────────────────┐
                          │     LLMFactory            │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌──────────────────────┐
              │ BaseLLMProvider     │       │ Settings (.env)      │
              └──────────┬──────────┘       └──────────────────────┘
                         │
     ┌───────────────────┼───────────────────┐
     ▼                   ▼                   ▼
┌──────────┐   ┌───────────────────┐   ┌────────────────────────┐
│ Mock     │   │ HuggingFace       │   │ OpenAI-Compatible      │
│ Provider │   │ Provider          │   │ Provider               │
└──────────┘   └───────────────────┘   └───────────┬────────────┘
                                                   │
                                     ┌─────────────┴────────────┐
                                     ▼                          ▼
                              ┌─────────────┐            ┌──────────────┐
                              │ vLLM/Ollama │            │ AI Kosh      │
                              └─────────────┘            └──────────────┘
```

---

## 1. Mock Provider (`LLM_PROVIDER=mock`)

- **Default for local development.**
- Requires **zero API keys**, **zero GPU memory**, and **no model downloads**.
- Synthesizes deterministic, clean answers directly from retrieved codebase chunks and personal notes.
- Guarantees zero latency and lightweight macOS execution.

### Configuration
```env
LLM_PROVIDER=mock
LLM_MODEL=mock-gpt-4o
```

---

## 2. HuggingFace Provider (`LLM_PROVIDER=huggingface`)

- Designed for local execution of open-weights models (e.g. `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3-8B-Instruct`).
- Encapsulates model parameter management, prompt formatting, tokenization, and device allocation (`cpu`, `cuda`, `mps`).
- Lazy architecture: raises `EchoMindException("Model not initialized.")` until a production inference engine (e.g., vLLM or `transformers`) is initialized.

### Configuration
```env
LLM_PROVIDER=huggingface
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
```

---

## 3. OpenAI-Compatible Provider (`LLM_PROVIDER=openai`)

- Standardized HTTP request payload and header builder compatible with any OpenAI API v1 protocol endpoint.
- Supports local inference servers (**vLLM**, **Ollama**, **LM Studio**) and cloud providers (**OpenAI**, **Groq**, **Together.ai**, **AI Kosh**).

### Configuration
```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=gpt-4o
LLM_API_KEY=your_api_key_here
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
```

---

## 4. AI Kosh Deployment Integration

AI Kosh exposes standard OpenAI-compatible `/v1/chat/completions` endpoints. To deploy EchoMind against AI Kosh, configure:

```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://aikosh.example.com/v1
LLM_MODEL=aikosh-qwen-72b
LLM_API_KEY=aikosh_sec_key_xyz
```

The `OpenAICompatibleProvider` builds standard authorization headers (`Authorization: Bearer <api_key>`) and payload structures expected by AI Kosh endpoints.

---

## Switching Providers

To switch providers in your environment, modify `.env` (or set environment variables):

```bash
# Switch to Mock Provider
export LLM_PROVIDER=mock

# Switch to OpenAI-Compatible Endpoint
export LLM_PROVIDER=openai
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=qwen2.5-7b

# Switch to HuggingFace Provider
export LLM_PROVIDER=huggingface
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

No code changes are required in `AskService` or the CLI tools when switching providers.
