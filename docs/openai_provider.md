# OpenAI-Compatible LLM Gateway

EchoMind includes a production-ready **OpenAI-Compatible LLM Gateway Provider** (`OpenAICompatibleProvider`), capable of connecting to any inference server or cloud deployment adhering to the OpenAI Chat Completions API standard (`/v1/chat/completions`).

---

## Overview & Architecture

The gateway abstracts HTTP request construction, headers, payloads, error handling, SSE streaming response parsing, and standard JSON response parsing without adding provider-specific logic to `AskService`.

```
                      ┌──────────────────────────┐
                      │    AskService (RAG)      │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │    BaseLLMProvider       │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │ OpenAICompatibleProvider │
                      └────────────┬─────────────┘
                                   │
 ┌──────────────┬──────────────┬───┴──────────┬──────────────┐
 ▼              ▼              ▼              ▼              ▼
OpenAI        vLLM         LM Studio       AI Kosh        OpenCode
Cloud        Local Server  Local App       Enterprise     Dev Endpoint
```

---

## Supported Endpoints & Configuration Examples

### 1. OpenAI Official Cloud API

Connect directly to OpenAI GPT models.

```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-proj-your-openai-api-key
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024
LLM_STREAM=false
```

### 2. vLLM High-Throughput Server

Connect to a locally hosted or GPU cluster vLLM inference server.

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_API_KEY=vllm-token
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
LLM_STREAM=false
```

### 3. LM Studio Desktop Local Server

Run local GGUF / Apple Silicon open-weights models via LM Studio.

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=qwen2.5-7b-instruct
LLM_API_KEY=lm-studio
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
LLM_STREAM=false
```

### 4. AI Kosh Enterprise Deployment

Connect to an AI Kosh managed enterprise deployment endpoint.

```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://aikosh.example.com/v1
LLM_MODEL=aikosh-qwen-72b-instruct
LLM_API_KEY=aikosh_secret_key_12345
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=1024
LLM_STREAM=false
```

### 5. OpenCode Development Server

Connect to an OpenCode developer server or proxy.

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=opencode-llama3-8b
LLM_API_KEY=opencode-dev
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
LLM_STREAM=false
```

---

## Token Streaming Support

`OpenAICompatibleProvider` implements `parse_stream_chunk(chunk_line: str)` and `generate_answer_stream()` for Server-Sent Events (SSE). Streaming tokens can be consumed by UIs or CLIs without modifying `AskService`.
