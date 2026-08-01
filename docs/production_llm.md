# Production LLM Integration & Verification Guide

This guide details how to configure, verify, and operate EchoMind with real OpenAI-compatible LLM endpoints (OpenAI, vLLM, LM Studio, AI Kosh, OpenCode).

---

## 1. Environment Configuration

EchoMind selects its LLM provider based on environment variables configured in `.env` or set at shell runtime.

### Configuration Parameters

| Environment Variable | Default Value | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | Active provider: `mock` or `openai` |
| `LLM_MODEL` | `mock-gpt-4o` | Target model identifier |
| `LLM_BASE_URL` | `http://localhost:8000/v1` | Base URL for OpenAI Chat Completions API |
| `LLM_API_KEY` | `None` | Bearer authorization token |
| `LLM_STREAM` | `false` | Enable SSE token streaming |
| `LLM_TEMPERATURE` | `0.7` | Generation sampling temperature |
| `LLM_MAX_TOKENS` | `512` | Maximum output token length |

---

## 2. Provider Endpoint Configurations

### A. OpenAI Official API

```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-proj-your-api-key-here
```

### B. vLLM High-Throughput Inference Server

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_API_KEY=vllm-token
```

### C. LM Studio Desktop Local Model

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=qwen2.5-7b-instruct
```

### D. AI Kosh Enterprise Managed Endpoint

```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://aikosh.example.com/v1
LLM_MODEL=aikosh-qwen-72b-instruct
LLM_API_KEY=aikosh_secret_key_12345
```

### E. OpenCode Local Server

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=opencode-llama3-8b
```

---

## 3. Verification Checklist

Follow this two-step verification workflow when connecting EchoMind to a new LLM endpoint.

### Step 1: Verification Probe CLI

Run the LLM verification CLI to test direct connectivity, latency, and model output:

```bash
python -m app.cli.verify_llm
```

#### Expected Output (Successful Connection):

```text
=================================================================
🤖 ECHOMIND PRODUCTION LLM GATEWAY VERIFICATION
=================================================================
  LLM Provider      : openai
  Model             : qwen2.5-7b-instruct
  Base URL          : http://localhost:8000/v1
  Streaming Enabled : False
  Temperature       : 0.7
  Max Tokens        : 512
=================================================================

Sending verification probe to endpoint: 'http://localhost:8000/v1'...

✅ Endpoint verification successful!

  Latency        : 142.50 ms
  Response Length: 5 chars (~1 words)
  Response Output:
  >>> READY
```

#### Expected Output (Unreachable Endpoint):

If the endpoint cannot be reached or returns an error status code, EchoMind **never falls back silently**. It raises a structured `EchoMindException`:

```text
❌ Verification failed with EchoMindException: Failed to connect to OpenAI-compatible endpoint 'http://localhost:8000/v1/chat/completions': [Errno 61] Connection refused
```

---

### Step 2: Full RAG Pipeline Question Answering

Once the endpoint is verified, execute end-to-end question answering via the `ask` CLI:

```bash
python -m app.cli.ask --query "What is the architecture of EchoMind?"
```

#### Expected Output:

```text
Question:
What is the architecture of EchoMind?

Answer:
EchoMind uses a hybrid RAG architecture combining dense vector embeddings with structural BM25-style keyword re-ranking, personal notes memory, and git commit history tracking...

Evidence:
backend/README.md (L1-L40)
notes/architecture.md (L1-L15)
```
