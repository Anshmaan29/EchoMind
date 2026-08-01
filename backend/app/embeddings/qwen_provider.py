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
    Intended for AI Kosh GPU environments (NVIDIA A100 / A100 MIG 3g.20gb).

    Features:
    - Automatic CUDA / MPS / CPU hardware detection
    - Hugging Face `device_map="cuda"` direct allocation to avoid NVML allocation crashes on MIG slices
    - bfloat16 mixed precision inference optimized for NVIDIA A100 GPUs
    - Dynamic model device discovery for input tensor placement
    - PyTorch autocast optimization
    - L2 Vector Normalization
    - Vector dimension validation (4096 dims default)
    - Hardware telemetry benchmarks (latency, throughput, peak VRAM, allocated/reserved memory)
    - STRICT EXCEPTION HANDLING: Never silently falls back to mock provider.
    """

    def __init__(self, model_name: str = None, dimension: int = 4096) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._dimension = dimension
        self.device = "cpu"
        self.torch_dtype = None
        self.model = None
        self.tokenizer = None
        self.model_device = None
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

            # Log CUDA memory before model load
            if self.device == "cuda" and torch.cuda.is_available():
                alloc_before = torch.cuda.memory_allocated(0) / (1024 * 1024)
                res_before = torch.cuda.memory_reserved(0) / (1024 * 1024)
                logger.info(
                    f"CUDA Memory Before Load: Allocated={alloc_before:.2f} MB, Reserved={res_before:.2f} MB"
                )

            logger.info(f"Loading Qwen Transformer Model '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
            }

            if self.device == "cuda":
                # Use Hugging Face direct CUDA placement to prevent CPU->GPU double-allocation NVML crashes on MIG slices
                model_kwargs["device_map"] = "cuda"
                model_kwargs["torch_dtype"] = self.torch_dtype
            elif self.torch_dtype is not None:
                model_kwargs["torch_dtype"] = self.torch_dtype

            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)

            if self.device != "cuda":
                self.model.to(self.device)

            self.model.eval()

            # Model device discovery for tokenized input tensor placement
            if hasattr(self.model, "device"):
                self.model_device = self.model.device
            else:
                try:
                    self.model_device = next(self.model.parameters()).device
                except Exception:
                    self.model_device = torch.device(self.device)

            self._is_initialized = True

            # Log CUDA memory after model load
            if self.device == "cuda" and torch.cuda.is_available():
                alloc_after = torch.cuda.memory_allocated(0) / (1024 * 1024)
                res_after = torch.cuda.memory_reserved(0) / (1024 * 1024)
                logger.info(
                    f"CUDA Memory After Load: Allocated={alloc_after:.2f} MB, Reserved={res_after:.2f} MB"
                )

            logger.info(
                f"QwenEmbeddingProvider successfully initialized '{self.model_name}' on device '{self.model_device}'."
            )

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

            target_device = self.model_device or getattr(
                self.model, "device", next(self.model.parameters()).device
            )

            # 1. Tokenize input batch and place on model's actual device
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(target_device)

            # 2. PyTorch Inference (bfloat16 autocast & no_grad optimization for A100)
            is_cuda = str(target_device).startswith("cuda") or self.device == "cuda"
            with torch.no_grad():
                if is_cuda:
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

            # 3. Telemetry Metrics Calculation
            batch_latency_ms = (time.perf_counter() - start_time) * 1000
            self.total_batches_processed += 1
            self.total_embeddings_generated += len(texts)
            self.total_latency_ms += batch_latency_ms

            if self.device == "cuda" and torch.cuda.is_available():
                peak_mem_bytes = torch.cuda.max_memory_allocated(0)
                self.peak_gpu_memory_mb = round(peak_mem_bytes / (1024 * 1024), 2)

            # 4. Dimension Verification
            results: list[list[float]] = []
            for vec in numpy_embeddings:
                vec_list = vec.tolist()
                if len(vec_list) != self._dimension:
                    raise EchoMindException(
                        f"Vector dimension mismatch for Qwen model: expected {self._dimension}, got {len(vec_list)}"
                    )
                results.append(vec_list)

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
            "device": str(self.model_device or self.device),
            "gpu_device_name": self.gpu_device_name,
            "model_name": self.model_name,
            "dimension": self._dimension,
            "total_batches_processed": self.total_batches_processed,
            "total_embeddings_generated": self.total_embeddings_generated,
            "average_batch_latency_ms": avg_latency,
            "embeddings_per_sec": throughput,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }
