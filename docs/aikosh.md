# AI Kosh GPU Environment & Heavy Embedding Guide

This guide details deploying **EchoMind** on **AI Kosh** (or another GPU cluster) to run high-throughput embedding generation using **Qwen/Qwen3-Embedding-8B** with CUDA and `bfloat16` precision.

---

## 🏛️ AI Kosh Architecture & Setup

On AI Kosh, heavy transformer embedding inference is executed using `QwenEmbeddingProvider`:
- Model: `Qwen/Qwen3-Embedding-8B` (1024 / 4096 dimensions)
- Precision: `bfloat16` (on NVIDIA A100 GPUs)
- Acceleration: PyTorch `torch.amp.autocast("cuda")` and `torch.no_grad()`

---

## 🚀 Step-by-Step AI Kosh Deployment Workflow

### 1. Upload & Clone Project
```bash
git clone https://github.com/Anshmaan29/EchoMind.git
cd EchoMind
```

### 2. Activate Python Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Configure GPU Environment Variables
Set `EMBEDDING_PROVIDER=qwen` in your `.env`:

```ini
EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-8B
EMBEDDING_DIMENSION=4096
```

### 4. Run GPU Hardware Verification Script
Verify that PyTorch detects your NVIDIA A100 GPU and loads `QwenEmbeddingProvider`:

```bash
python backend/scripts/verify_qwen_gpu.py
```

#### Verification Output Example:
```text
=================================================================
🚀 AI KOSH GPU VERIFICATION SCRIPT: QWEN EMBEDDING ENGINE
=================================================================
Target Model Name   : Qwen/Qwen3-Embedding-8B
Configured Dimension : 4096

📊 HARDWARE TELEMETRY
-----------------------------------------------------------------
CUDA Available     : True
Active Device      : CUDA
GPU Hardware Name  : NVIDIA A100-SXM4-80GB
PyTorch Dtype      : torch.bfloat16
Embedding Dimension: 4096
-----------------------------------------------------------------
✅ AI KOSH GPU VERIFICATION SUCCESSFUL
```

---

## ⚡ High-Throughput GPU Repository Embedding

Run batch embedding over target repository data:

```bash
PYTHONPATH=backend python -m app.cli.embed --input ./data --batch-size 64 --workers 4
```

### Generated Artifacts
- **Qdrant Vector Index**: Populates target collection `echomind_memories`.
- **Local Backup Stream**: Appends full payloads to `data/embeddings_backup.jsonl`.
- **Checkpoint Ledger**: Recorded in `.checkpoints/embedding_checkpoint.db`.

---

## 📥 Downloading Results

Download the generated vector payloads and backups back to your workspace:
```bash
scp user@aikosh:/path/to/EchoMind/data/embeddings_backup.jsonl ./data/
```
