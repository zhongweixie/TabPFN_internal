#!/usr/bin/env python3
#  Copyright (c) Prior Labs GmbH 2026.
"""
Experiment 1: Basic Classification Test

Tests TabPFN-3 on standard classification datasets to verify basic functionality
and reproduce baseline performance metrics.

Based on paper Section 3.1 (Public Tabular Benchmarks)
"""

import os
os.environ["TABPFN_NO_BROWSER"] = "1"  # Disable browser authentication

import sys
sys.path.append('..')

from sklearn.datasets import load_breast_cancer, load_wine, load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tabpfn import TabPFNClassifier
import torch

from utils import (
    print_header, print_section, print_result,
    MemoryTracker, Timer, BenchmarkResult,
    save_results_to_csv, create_summary_table,
    print_system_info
)


def test_binary_classification():
    """Test on binary classification dataset (Breast Cancer)"""
    print_section("Binary Classification - Breast Cancer Dataset")

    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    print_result("Train samples", X_train.shape[0])
    print_result("Test samples", X_test.shape[0])
    print_result("Features", X_train.shape[1])
    print_result("Classes", len(set(y)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabPFNClassifier(device=device)

    # Fit
    with MemoryTracker(device) as mem:
        with Timer() as fit_timer:
            clf.fit(X_train, y_train)
        mem_info = mem.get_peak_memory()

    # Predict
    with Timer() as pred_timer:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    f1 = f1_score(y_test, y_pred)

    print_result("Accuracy", f"{acc:.4f}")
    print_result("ROC AUC", f"{roc_auc:.4f}")
    print_result("F1 Score", f"{f1:.4f}")
    print_result("Fit time", f"{fit_timer.elapsed:.3f}s")
    print_result("Predict time", f"{pred_timer.elapsed:.3f}s")
    print_result("Peak CPU memory", f"{mem_info['cpu_mb']:.1f} MB")
    print_result("Peak GPU memory", f"{mem_info['gpu_mb']:.1f} MB")

    return BenchmarkResult(
        experiment_name="binary_classification",
        model_name="TabPFN-3",
        dataset_size=(X_train.shape[0], X_train.shape[1]),
        n_classes=2,
        fit_time=fit_timer.elapsed,
        predict_time=pred_timer.elapsed,
        total_time=fit_timer.elapsed + pred_timer.elapsed,
        metric_value=acc,
        metric_name="accuracy",
        memory_peak_mb=mem_info['cpu_mb'],
        gpu_memory_mb=mem_info['gpu_mb'],
        additional_info={"roc_auc": roc_auc, "f1_score": f1}
    )


def test_multiclass_classification():
    """Test on multiclass classification dataset (Wine)"""
    print_section("Multiclass Classification - Wine Dataset")

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    print_result("Train samples", X_train.shape[0])
    print_result("Test samples", X_test.shape[0])
    print_result("Features", X_train.shape[1])
    print_result("Classes", len(set(y)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/home/zxiebk/workspace/model/tabpfn_3/tabpfn-v3-classifier-v3_default.ckpt"
    clf = TabPFNClassifier(device=device, model_path=model_path)

    with MemoryTracker(device) as mem:
        with Timer() as fit_timer:
            clf.fit(X_train, y_train)
        mem_info = mem.get_peak_memory()

    with Timer() as pred_timer:
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print_result("Accuracy", f"{acc:.4f}")
    print_result("F1 Score (macro)", f"{f1:.4f}")
    print_result("Fit time", f"{fit_timer.elapsed:.3f}s")
    print_result("Predict time", f"{pred_timer.elapsed:.3f}s")

    return BenchmarkResult(
        experiment_name="multiclass_classification",
        model_name="TabPFN-3",
        dataset_size=(X_train.shape[0], X_train.shape[1]),
        n_classes=len(set(y)),
        fit_time=fit_timer.elapsed,
        predict_time=pred_timer.elapsed,
        total_time=fit_timer.elapsed + pred_timer.elapsed,
        metric_value=acc,
        metric_name="accuracy",
        memory_peak_mb=mem_info['cpu_mb'],
        gpu_memory_mb=mem_info['gpu_mb'],
        additional_info={"f1_macro": f1}
    )


def test_medium_dataset():
    """Test on medium-sized synthetic dataset"""
    print_section("Medium Dataset - Synthetic (10K samples)")

    X, y = make_classification(
        n_samples=10000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=5,
        n_clusters_per_class=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print_result("Train samples", X_train.shape[0])
    print_result("Test samples", X_test.shape[0])
    print_result("Features", X_train.shape[1])
    print_result("Classes", len(set(y)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/home/zxiebk/workspace/model/tabpfn_3/tabpfn-v3-classifier-v3_default.ckpt"
    clf = TabPFNClassifier(n_estimators=4, device=device, model_path=model_path)

    with MemoryTracker(device) as mem:
        with Timer() as fit_timer:
            clf.fit(X_train, y_train)
        mem_info = mem.get_peak_memory()

    with Timer() as pred_timer:
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print_result("Accuracy", f"{acc:.4f}")
    print_result("Fit time", f"{fit_timer.elapsed:.3f}s")
    print_result("Predict time", f"{pred_timer.elapsed:.3f}s")
    print_result("Peak GPU memory", f"{mem_info['gpu_mb']:.1f} MB")

    return BenchmarkResult(
        experiment_name="medium_dataset",
        model_name="TabPFN-3",
        dataset_size=(X_train.shape[0], X_train.shape[1]),
        n_classes=5,
        fit_time=fit_timer.elapsed,
        predict_time=pred_timer.elapsed,
        total_time=fit_timer.elapsed + pred_timer.elapsed,
        metric_value=acc,
        metric_name="accuracy",
        memory_peak_mb=mem_info['cpu_mb'],
        gpu_memory_mb=mem_info['gpu_mb']
    )


def main():
    print_header("Experiment 1: Basic Classification")
    print_system_info()

    results = []

    # Test 1: Binary classification
    results.append(test_binary_classification())

    # Test 2: Multiclass classification
    results.append(test_multiclass_classification())

    # Test 3: Medium dataset
    results.append(test_medium_dataset())

    # Summary
    print_header("Summary")
    summary = create_summary_table(results)
    print(summary.to_string(index=False))

    # Save results
    save_results_to_csv(results, "results_01_basic_classification.csv")

    print("\n✓ Experiment 1 completed successfully!")


if __name__ == "__main__":
    main()
