#!/usr/bin/env python3
#  Copyright (c) Prior Labs GmbH 2026.
"""
Experiment 4: KV Cache Speedup Test

Reproduces the KV cache performance improvements shown in Figure 7-8 of the paper.
Tests the speedup from using fit_mode="fit_with_cache" vs "fit_preprocessors".

Expected results (from paper Section 2.4.2):
- 1-3 orders of magnitude speedup for cached prediction
- KV cache size: ~7GB per estimator @ 1M rows
- Latency: 0.1-3 ms/test point @ batch=100
"""

import sys
sys.path.append('..')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
import torch
import numpy as np

from utils import (
    print_header, print_section, print_result,
    MemoryTracker, Timer, BenchmarkResult,
    save_results_to_csv, format_time,
    print_system_info
)


def test_kv_cache_speedup(n_train: int, n_features: int, n_test: int = 100):
    """
    Test KV cache speedup for a given dataset size.

    Args:
        n_train: Number of training samples
        n_features: Number of features
        n_test: Number of test samples
    """
    print_section(f"KV Cache Test: {n_train:,} train × {n_features} features, {n_test} test")

    # Generate data
    X, y = make_classification(
        n_samples=n_train + n_test,
        n_features=n_features,
        n_informative=min(n_features, 50),
        n_redundant=min(n_features // 5, 10),
        n_classes=min(10, n_train // 100),
        random_state=42
    )
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print_result("Train samples", f"{n_train:,}")
    print_result("Test samples", f"{n_test:,}")
    print_result("Features", n_features)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    # ====================================================================
    # Mode 1: fit_preprocessors (no KV cache)
    # ====================================================================
    print("\n[Mode 1] fit_preprocessors (no KV cache)")

    clf_no_cache = TabPFNClassifier(
        fit_mode="fit_preprocessors",
        n_estimators=1,  # Use 1 estimator for clearer measurement
        device=device
    )

    with MemoryTracker(device) as mem:
        with Timer() as fit_timer:
            clf_no_cache.fit(X_train, y_train)
        mem_no_cache = mem.get_peak_memory()

    # Multiple predictions to measure average time
    pred_times = []
    for _ in range(3):
        with Timer() as pred_timer:
            y_pred = clf_no_cache.predict(X_test)
        pred_times.append(pred_timer.elapsed)

    pred_time_no_cache = np.mean(pred_times)
    acc_no_cache = accuracy_score(y_test, y_pred)

    print_result("Fit time", format_time(fit_timer.elapsed))
    print_result("Predict time (avg)", format_time(pred_time_no_cache))
    print_result("Accuracy", f"{acc_no_cache:.4f}")
    print_result("Peak GPU memory", f"{mem_no_cache['gpu_mb']:.1f} MB")

    results.append(BenchmarkResult(
        experiment_name=f"kv_cache_{n_train}",
        model_name="no_cache",
        dataset_size=(n_train, n_features),
        n_classes=len(set(y_train)),
        fit_time=fit_timer.elapsed,
        predict_time=pred_time_no_cache,
        total_time=fit_timer.elapsed + pred_time_no_cache,
        metric_value=acc_no_cache,
        metric_name="accuracy",
        memory_peak_mb=mem_no_cache['cpu_mb'],
        gpu_memory_mb=mem_no_cache['gpu_mb'],
        additional_info={"n_test": n_test, "ms_per_sample": pred_time_no_cache * 1000 / n_test}
    ))

    # ====================================================================
    # Mode 2: fit_with_cache (KV cache enabled)
    # ====================================================================
    print("\n[Mode 2] fit_with_cache (KV cache enabled)")

    clf_cache = TabPFNClassifier(
        fit_mode="fit_with_cache",
        n_estimators=1,
        device=device
    )

    with MemoryTracker(device) as mem:
        with Timer() as fit_timer:
            clf_cache.fit(X_train, y_train)
        mem_cache = mem.get_peak_memory()

    # Multiple predictions to measure average cached time
    pred_times = []
    for _ in range(3):
        with Timer() as pred_timer:
            y_pred = clf_cache.predict(X_test)
        pred_times.append(pred_timer.elapsed)

    pred_time_cache = np.mean(pred_times)
    acc_cache = accuracy_score(y_test, y_pred)

    print_result("Fit (build cache) time", format_time(fit_timer.elapsed))
    print_result("Predict (cached) time (avg)", format_time(pred_time_cache))
    print_result("Accuracy", f"{acc_cache:.4f}")
    print_result("Peak GPU memory", f"{mem_cache['gpu_mb']:.1f} MB")
    print_result("KV cache size (estimated)", f"{mem_cache['gpu_mb'] - mem_no_cache['gpu_mb']:.1f} MB")

    results.append(BenchmarkResult(
        experiment_name=f"kv_cache_{n_train}",
        model_name="with_cache",
        dataset_size=(n_train, n_features),
        n_classes=len(set(y_train)),
        fit_time=fit_timer.elapsed,
        predict_time=pred_time_cache,
        total_time=fit_timer.elapsed + pred_time_cache,
        metric_value=acc_cache,
        metric_name="accuracy",
        memory_peak_mb=mem_cache['cpu_mb'],
        gpu_memory_mb=mem_cache['gpu_mb'],
        additional_info={"n_test": n_test, "ms_per_sample": pred_time_cache * 1000 / n_test}
    ))

    # ====================================================================
    # Comparison
    # ====================================================================
    print("\n[Comparison]")
    speedup = pred_time_no_cache / pred_time_cache
    fit_overhead = (fit_timer.elapsed - fit_timer.elapsed) / fit_timer.elapsed if fit_timer.elapsed > 0 else 0

    print_result("Prediction speedup", f"{speedup:.1f}x")
    print_result("ms per test sample (no cache)", f"{pred_time_no_cache * 1000 / n_test:.2f}")
    print_result("ms per test sample (cached)", f"{pred_time_cache * 1000 / n_test:.2f}")
    print_result("Memory overhead", f"{mem_cache['gpu_mb'] - mem_no_cache['gpu_mb']:.1f} MB")

    return results, speedup


def main():
    print_header("Experiment 4: KV Cache Speedup Test")
    print_system_info()

    all_results = []
    speedups = []

    # Test configurations based on paper Figure 7-8
    # (Scaled down for feasibility)
    test_configs = [
        # (n_train, n_features, n_test)
        (5000, 10, 100),      # Small
        (10000, 50, 100),     # Medium
        (50000, 100, 100),    # Large (from paper examples)
    ]

    # Add even larger test if enough GPU memory
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30e9:
        test_configs.append((100000, 100, 100))  # Very large

    for n_train, n_features, n_test in test_configs:
        results, speedup = test_kv_cache_speedup(n_train, n_features, n_test)
        all_results.extend(results)
        speedups.append((n_train, speedup))

    # Summary
    print_header("Summary: KV Cache Speedup")
    print("\nSpeedup by dataset size:")
    for n_train, speedup in speedups:
        print_result(f"{n_train:,} samples", f"{speedup:.1f}x")

    avg_speedup = np.mean([s for _, s in speedups])
    print_result("\nAverage speedup", f"{avg_speedup:.1f}x")

    # Save results
    save_results_to_csv(all_results, "results_04_kv_cache_speedup.csv")

    print("\n" + "=" * 80)
    print("Expected from paper (Section 2.4.2):")
    print("  - Speedup: 1-3 orders of magnitude (10x-1000x)")
    print("  - Latency: 0.1-3 ms/test point @ batch=100")
    print("  - KV cache: ~7GB per estimator @ 1M rows")
    print("=" * 80)

    print("\n✓ Experiment 4 completed successfully!")


if __name__ == "__main__":
    main()
