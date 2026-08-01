#!/usr/bin/env python3
"""
AI Kosh GPU Verification Script for QwenEmbeddingProvider.
Usage:
    python backend/scripts/verify_qwen_gpu.py

This script initializes QwenEmbeddingProvider on GPU, verifies CUDA availability,
inspects precision dtypes, generates embeddings for sample queries, and prints
pairwise cosine similarity benchmarks.
"""

import asyncio
import os
import sys
import numpy as np

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.embeddings.qwen_provider import QwenEmbeddingProvider

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

async def main() -> None:
    print("\n" + "=" * 65)
    print("🚀 AI KOSH GPU VERIFICATION SCRIPT: QWEN EMBEDDING ENGINE")
    print("=" * 65)

    model_name = settings.EMBEDDING_MODEL_NAME

    try:
        provider = QwenEmbeddingProvider(model_name=model_name)
    except Exception as e:
        print(f"\n❌ FAILED to load QwenEmbeddingProvider: {e}")
        print("Check CUDA drivers, PyTorch installation, or Hugging Face access.\n")
        sys.exit(1)

    import torch

    cuda_available = torch.cuda.is_available()
    device = provider.device
    gpu_name = provider.gpu_device_name
    dtype = str(provider.torch_dtype)

    print("\n📊 HARDWARE TELEMETRY")
    print("-" * 65)
    print(f"CUDA Available     : {cuda_available}")
    print(f"Active Device      : {device.upper()}")
    print(f"GPU Hardware Name  : {gpu_name}")
    print(f"PyTorch Dtype      : {dtype}")
    print(f"Embedding Dimension: {provider.dimension}")
    print("-" * 65)

    sample_texts = [
        "EchoMind",
        "FastAPI",
        "Qwen embeddings"
    ]

    print("\n⚡ GENERATING TEST EMBEDDINGS...")
    embeddings = await provider.embed_texts(sample_texts)

    print(f"Generated {len(embeddings)} embedding vectors.")
    for idx, (txt, vec) in enumerate(zip(sample_texts, embeddings)):
        norm = np.linalg.norm(vec)
        print(f"  [{idx+1}] Text: \"{txt:<18}\" | Dims: {len(vec)} | L2 Norm: {norm:.4f}")

    print("\n📐 PAIRWISE COSINE SIMILARITIES")
    print("-" * 65)
    sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])
    sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])
    sim_1_2 = cosine_similarity(embeddings[1], embeddings[2])

    print(f"  • Sim(\"{sample_texts[0]}\", \"{sample_texts[1]}\") : {sim_0_1:.4f}")
    print(f"  • Sim(\"{sample_texts[0]}\", \"{sample_texts[2]}\") : {sim_0_2:.4f}")
    print(f"  • Sim(\"{sample_texts[1]}\", \"{sample_texts[2]}\") : {sim_1_2:.4f}")
    print("-" * 65)

    benchmarks = provider.get_benchmarks()
    print("\n📈 BENCHMARK SUMMARY")
    print("-" * 65)
    print(f"Average Batch Latency : {benchmarks['average_batch_latency_ms']} ms")
    print(f"Throughput            : {benchmarks['embeddings_per_sec']} vec/sec")
    if benchmarks['peak_gpu_memory_mb'] > 0:
        print(f"Peak GPU VRAM         : {benchmarks['peak_gpu_memory_mb']} MB")
    print("=" * 65)
    print("✅ AI KOSH GPU VERIFICATION SUCCESSFUL\n")

if __name__ == "__main__":
    asyncio.run(main())
