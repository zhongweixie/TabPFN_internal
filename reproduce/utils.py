#!/usr/bin/env python3
#  Copyright (c) Prior Labs GmbH 2026.
"""
Utility functions for TabPFN-3 experiment reproduction.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    experiment_name: str
    model_name: str
    dataset_size: Tuple[int, int]  # (n_samples, n_features)
    n_classes: int
    fit_time: float
    predict_time: float
    total_time: float
    metric_value: float
    metric_name: str
    memory_peak_mb: float
    gpu_memory_mb: float = 0.0
    additional_info: Dict[str, Any] = None


class MemoryTracker:
    """Track CPU and GPU memory usage"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.start_cpu_mem = 0
        self.start_gpu_mem = 0

    def __enter__(self):
        self.start_cpu_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage"""
        cpu_mem = psutil.Process().memory_info().rss / 1024 / 1024 - self.start_cpu_mem
        gpu_mem = 0.0
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        return {"cpu_mb": cpu_mem, "gpu_mb": gpu_mem}


class Timer:
    """Simple timer context manager"""

    def __init__(self):
        self.start_time = 0
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time


def print_header(title: str, width: int = 80):
    """Print formatted header"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_section(title: str, width: int = 80):
    """Print formatted section"""
    print("\n" + title)
    print("-" * width)


def print_result(label: str, value: Any, unit: str = ""):
    """Print formatted result"""
    print(f"  {label:30s}: {value}{unit}")


def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "current_device": torch.cuda.current_device()
    }


def print_system_info():
    """Print system information"""
    print_header("System Information")

    # PyTorch info
    print_result("PyTorch version", torch.__version__)
    print_result("Python version", f"{psutil.sys.version.split()[0]}")

    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print_result("CUDA available", "Yes")
        print_result("GPU", gpu_info["device_name"])
        print_result("GPU count", gpu_info["device_count"])
        print_result("GPU memory", f"{gpu_info['total_memory_gb']:.1f} GB")
    else:
        print_result("CUDA available", "No")

    # CPU info
    print_result("CPU count", psutil.cpu_count())
    print_result("RAM", f"{psutil.virtual_memory().total / 1e9:.1f} GB")


def save_results_to_csv(results: List[BenchmarkResult], filename: str):
    """Save benchmark results to CSV"""
    data = []
    for r in results:
        row = {
            "experiment": r.experiment_name,
            "model": r.model_name,
            "n_samples": r.dataset_size[0],
            "n_features": r.dataset_size[1],
            "n_classes": r.n_classes,
            "fit_time_s": r.fit_time,
            "predict_time_s": r.predict_time,
            "total_time_s": r.total_time,
            f"{r.metric_name}": r.metric_value,
            "peak_memory_mb": r.memory_peak_mb,
            "gpu_memory_mb": r.gpu_memory_mb
        }
        if r.additional_info:
            row.update(r.additional_info)
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n✓ Results saved to: {filename}")


def create_summary_table(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Create summary table from results"""
    data = []
    for r in results:
        data.append({
            "Experiment": r.experiment_name,
            "Model": r.model_name,
            "Dataset": f"{r.dataset_size[0]}×{r.dataset_size[1]}",
            "Fit Time": format_time(r.fit_time),
            "Predict Time": format_time(r.predict_time),
            r.metric_name: f"{r.metric_value:.4f}",
            "Memory (GB)": f"{r.memory_peak_mb/1024:.2f}"
        })
    return pd.DataFrame(data)


def compare_models(results: List[BenchmarkResult],
                   baseline_model: str,
                   metric: str = "total_time") -> pd.DataFrame:
    """Compare models against baseline"""
    baseline = [r for r in results if r.model_name == baseline_model]
    if not baseline:
        print(f"Warning: Baseline model '{baseline_model}' not found")
        return pd.DataFrame()

    baseline_dict = {r.experiment_name: r for r in baseline}

    comparison = []
    for r in results:
        if r.model_name == baseline_model:
            continue

        base = baseline_dict.get(r.experiment_name)
        if not base:
            continue

        if metric == "total_time":
            speedup = base.total_time / r.total_time
        elif metric == "fit_time":
            speedup = base.fit_time / r.fit_time
        elif metric == "predict_time":
            speedup = base.predict_time / r.predict_time
        else:
            speedup = 0.0

        comparison.append({
            "Experiment": r.experiment_name,
            "Model": r.model_name,
            "Baseline": baseline_model,
            "Speedup": f"{speedup:.2f}x",
            "Metric": metric
        })

    return pd.DataFrame(comparison)
