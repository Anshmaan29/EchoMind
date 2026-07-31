import json
import logging
import os
import sys
import time
from typing import Any
import yaml

# Add backend directory to sys.path to enable direct reuse of EchoMind modules
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

class ExperimentHarness:
    """
    Base Harness for AI Kosh GPU Experiments.
    Loads YAML configs, sets up logging, detects CUDA hardware, manages checkpoints, and prints telemetry.
    """
    def __init__(self, config_path: str = None, experiment_name: str = "gpu_experiment") -> None:
        self.experiment_name = experiment_name
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.config_path = config_path or os.path.join(self.root_dir, "experiments", "configs", "default_config.yaml")
        self.config = self._load_config()

        # Paths
        self.output_dir = os.path.join(self.root_dir, "experiments", "outputs")
        self.log_dir = os.path.join(self.root_dir, "experiments", "logs")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._setup_logger()
        self.device, self.gpu_name = self._detect_hardware()

    def _load_config(self) -> dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "experiment": {"batch_size": 64, "max_workers": 4, "precision": "float16", "resume": True},
            "model": {"name": "BAAI/bge-m3", "dimension": 1024}
        }

    def _setup_logger(self) -> None:
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.info(f"Initialized Experiment logger at '{log_file}'.")

    def _detect_hardware(self) -> tuple[str, str]:
        device = "cpu"
        gpu_name = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
            elif torch.backends.mps.is_available():
                device = "mps"
                gpu_name = "Apple Silicon MPS"
        except Exception:
            pass

        self.logger.info(f"Auto-detected Hardware Device: '{device}' ({gpu_name}).")
        return device, gpu_name

    def save_output_jsonl(self, filename: str, records: list[dict[str, Any]]) -> str:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        self.logger.info(f"Saved {len(records)} records to output JSONL '{filepath}'.")
        return filepath

    def print_benchmark_summary(
        self,
        total_items: int,
        processed_items: int,
        elapsed_seconds: float,
        extra_metrics: dict[str, Any] = None
    ) -> None:
        elapsed = max(0.001, elapsed_seconds)
        throughput = round(processed_items / elapsed, 2)

        peak_vram_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
        except Exception:
            pass

        print("\n" + "=" * 65)
        print(f"📊 AI KOSH GPU EXPERIMENT SUMMARY: {self.experiment_name.upper()}")
        print("=" * 65)
        print(f"Hardware Accelerator   : {self.device.upper()} ({self.gpu_name})")
        print(f"Total Evaluated Items  : {total_items}")
        print(f"Processed Items        : {processed_items}")
        print(f"Elapsed Time           : {round(elapsed, 2)} s")
        print(f"Throughput             : {throughput} items/sec")
        if peak_vram_mb > 0:
            print(f"Peak GPU VRAM Memory   : {peak_vram_mb} MB")
        if extra_metrics:
            for k, v in extra_metrics.items():
                print(f"{k:<23} : {v}")
        print("=" * 65 + "\n")
