import time
from typing import Any
import numpy as np
from app.core.config import settings
from app.core.exceptions import EchoMindException
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingProvider


class QwenEmbeddingProvider(BaseEmbeddingProvider):
    """
    Production Embedding Provider using Qwen/Qwen3-Embedding-8B via PyTorch & Transformers.
    Intended for AI Kosh GPU environments.

    Features:
    - Automatic CUDA / MPS / CPU hardware detection
    - Standard single-GPU model loading (AutoModel.from_pretrained + self.model.cuda())
    - bfloat16 mixed precision inference optimized for NVIDIA A100 GPUs
    - torch.inference_mode() memory-efficient inference
    - Post-batch CUDA tensor deletion, synchronization, and empty_cache memory cleanup
    - Detailed GPU memory telemetry per batch (allocated, reserved, peak MB)
    - L2 Vector Normalization
    - Vector dimension validation (4096 dims default)
    - Hardware telemetry benchmarks (latency, throughput, peak VRAM)
    - STRICT EXCEPTION HANDLING: Never silently falls back to mock provider.
    """

    def __init__(self, model_name: str = None, dimension: int = 4096) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._dimension = dimension
        self.device = "cpu"
        self.torch_dtype = None
        self.model = None
        self.tokenizer = None
        self._is_initialized = False

        # Telemetry metrics
        self.total_batches_processed = 0
        self.total_embeddings_generated = 0
        self.total_latency_ms = 0.0
        self.peak_gpu_memory_mb = 0.0
        self.gpu_device_name = "N/A"

        self._initialize_device_and_model()

    def _initialize_device_and_model(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if torch.cuda.is_available():
                self.device = "cuda"
                self.gpu_device_name = torch.cuda.get_device_name(0)
                if torch.cuda.is_bf16_supported():
                    self.torch_dtype = torch.bfloat16
                else:
                    self.torch_dtype = torch.float16
                logger.info(
                    f"QwenEmbeddingProvider using CUDA GPU: '{self.gpu_device_name}' with precision {self.torch_dtype}."
                )
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.gpu_device_name = "Apple Silicon MPS"
                self.torch_dtype = torch.float32
                logger.info("QwenEmbeddingProvider using Apple Silicon MPS acceleration.")
            else:
                self.device = "cpu"
                self.gpu_device_name = "CPU"
                self.torch_dtype = torch.float32
                logger.info("QwenEmbeddingProvider running on CPU.")

            logger.info(f"Loading Qwen Transformer Model '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype if self.device == "cuda" else None,
            )

            if self.device == "cuda":
                self.model.cuda()
            elif self.device == "mps":
                self.model.to("mps")
            else:
                self.model.to("cpu")

            self.model.eval()
            self._is_initialized = True
            logger.info(f"QwenEmbeddingProvider successfully initialized '{self.model_name}'.")

        except Exception as e:
            logger.error(f"CRITICAL: Failed to load Qwen embedding model '{self.model_name}': {e}")
            self._is_initialized = False
            raise EchoMindException(
                f"Failed to initialize QwenEmbeddingProvider with model '{self.model_name}': {e}"
            ) from e

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if not self._is_initialized or self.model is None:
            raise EchoMindException(
                f"QwenEmbeddingProvider model '{self.model_name}' is not initialized."
            )

        start_time = time.perf_counter()

        try:
            import torch

            # 1. Tokenize input batch
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # 2. PyTorch Inference using torch.inference_mode() for minimal memory overhead
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.amp.autocast("cuda", dtype=self.torch_dtype):
                        model_output = self.model(**encoded_input)
                else:
                    model_output = self.model(**encoded_input)

                # Mean pooling over hidden states
                attention_mask = encoded_input['attention_mask'].unsqueeze(-1).to(torch.float32)
                token_embeddings = model_output[0].to(torch.float32)
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                sentence_embeddings = sum_embeddings / sum_mask

                # L2 Normalization before returning
                normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                numpy_embeddings = normalized_embeddings.cpu().numpy()

            # 3. Dimension Verification
            results: list[list[float]] = []
            for vec in numpy_embeddings:
                vec_list = vec.tolist()
                if len(vec_list) != self._dimension:
                    raise EchoMindException(
                        f"Vector dimension mismatch for Qwen model: expected {self._dimension}, got {len(vec_list)}"
                    )
                results.append(vec_list)

            # 4. Telemetry Metrics Calculation
            batch_latency_ms = (time.perf_counter() - start_time) * 1000
            self.total_batches_processed += 1
            self.total_embeddings_generated += len(texts)
            self.total_latency_ms += batch_latency_ms

            # 5. Explicitly delete batch intermediate tensors to release PyTorch allocator references
            del encoded_input
            del model_output
            del token_embeddings
            del attention_mask
            del sum_embeddings
            del sum_mask
            del sentence_embeddings
            del normalized_embeddings
            del numpy_embeddings

            # 6. Post-batch CUDA synchronization & cache clearing (after batch completes)
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
                res_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
                peak_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
                self.peak_gpu_memory_mb = peak_mb

                logger.info(
                    f"Qwen Batch {self.total_batches_processed} ({len(texts)} items, {batch_latency_ms:.1f}ms) | "
                    f"GPU Memory: allocated={alloc_mb}MB, reserved={res_mb}MB, peak={peak_mb}MB"
                )

            return results

        except Exception as e:
            logger.error(f"Error during Qwen embedding inference: {e}")
            raise EchoMindException(f"QwenEmbeddingProvider inference failed: {e}") from e

    def get_benchmarks(self) -> dict[str, Any]:
        """Returns real-time inference and hardware telemetry metrics."""
        avg_latency = (
            round(self.total_latency_ms / self.total_batches_processed, 2)
            if self.total_batches_processed > 0 else 0.0
        )
        throughput = (
            round(self.total_embeddings_generated / (self.total_latency_ms / 1000), 2)
            if self.total_latency_ms > 0 else 0.0
        )

        return {
            "device": self.device,
            "gpu_device_name": self.gpu_device_name,
            "model_name": self.model_name,
            "dimension": self._dimension,
            "total_batches_processed": self.total_batches_processed,
            "total_embeddings_generated": self.total_embeddings_generated,
            "average_batch_latency_ms": avg_latency,
            "embeddings_per_sec": throughput,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }
