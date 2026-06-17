#!/usr/bin/env python3
#  Copyright (c) Prior Labs GmbH 2026.
"""
Complete TabPFN-3 Experiment Suite

Reproduces key experiments from the paper without requiring authentication.
"""

import os
import time
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, r2_score
from tabpfn import TabPFNClassifier, TabPFNRegressor
import torch

# Configuration
MODEL_PATH_CLASSIFIER = "/home/zxiebk/workspace/model/tabpfn_3/tabpfn-v3-classifier-v3_default.ckpt"
MODEL_PATH_REGRESSOR = "/home/zxiebk/workspace/model/tabpfn_3/tabpfn-v3-regressor-v3_default.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_header(title):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

def print_section(title):
    print(f"\n{title}")
    print("-" * 80)

def print_result(label, value):
    print(f"  {label:30s}: {value}")

# ============================================================================
# Experiment 1: Binary Classification
# ============================================================================
def test_binary_classification():
    print_section("Experiment 1: Binary Classification (Breast Cancer)")

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print_result("Train samples", X_train.shape[0])
    print_result("Test samples", X_test.shape[0])
    print_result("Features", X_train.shape[1])

    clf = TabPFNClassifier(device=DEVICE, model_path=MODEL_PATH_CLASSIFIER)

    start = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    pred_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    f1 = f1_score(y_test, y_pred)

    print_result("Accuracy", f"{acc:.4f}")
    print_result("ROC AUC", f"{roc_auc:.4f}")
    print_result("F1 Score", f"{f1:.4f}")
    print_result("Fit time", f"{fit_time:.3f}s")
    print_result("Predict time", f"{pred_time:.3f}s")
    print_result("Total time", f"{fit_time + pred_time:.3f}s")

    return {"acc": acc, "roc_auc": roc_auc, "f1": f1, "fit_time": fit_time, "pred_time": pred_time}

# ============================================================================
# Experiment 2: Multiclass Classification
# ============================================================================
def test_multiclass_classification():
    print_section("Experiment 2: Multiclass Classification (Wine)")

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print_result("Train samples", X_train.shape[0])
    print_result("Test samples", X_test.shape[0])
    print_result("Features", X_train.shape[1])
    print_result("Classes", len(set(y)))

    clf = TabPFNClassifier(device=DEVICE, model_path=MODEL_PATH_CLASSIFIER)

    start = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print_result("Accuracy", f"{acc:.4f}")
    print_result("F1 Score (macro)", f"{f1:.4f}")
    print_result("Fit time", f"{fit_time:.3f}s")
    print_result("Predict time", f"{pred_time:.3f}s")

    return {"acc": acc, "f1": f1, "fit_time": fit_time, "pred_time": pred_time}

# ============================================================================
# Experiment 3: Many-Class Classification
# ============================================================================
def test_many_class_classification():
    print_section("Experiment 3: Many-Class Classification (50 classes)")

    X, y = make_classification(
        n_samples=10000, n_features=50, n_informative=30,
        n_classes=50, n_clusters_per_class=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print_result("Train samples", X_train.shape[0])
    print_result("Test samples", X_test.shape[0])
    print_result("Features", X_train.shape[1])
    print_result("Classes", len(set(y)))

    clf = TabPFNClassifier(n_estimators=4, device=DEVICE, model_path=MODEL_PATH_CLASSIFIER)

    start = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)

    print_result("Accuracy", f"{acc:.4f}")
    print_result("Fit time", f"{fit_time:.3f}s")
    print_result("Predict time", f"{pred_time:.3f}s")

    return {"acc": acc, "fit_time": fit_time, "pred_time": pred_time}

# ============================================================================
# Experiment 4: Regression
# ============================================================================
def test_regression():
    print_section("Experiment 4: Regression")

    X, y = make_regression(n_samples=5000, n_features=20, n_informative=15, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print_result("Train samples", X_train.shape[0])
    print_result("Test samples", X_test.shape[0])
    print_result("Features", X_train.shape[1])

    reg = TabPFNRegressor(device=DEVICE, model_path=MODEL_PATH_REGRESSOR)

    start = time.time()
    reg.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = reg.predict(X_test)
    pred_time = time.time() - start

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print_result("MSE", f"{mse:.4f}")
    print_result("R2 Score", f"{r2:.4f}")
    print_result("Fit time", f"{fit_time:.3f}s")
    print_result("Predict time", f"{pred_time:.3f}s")

    return {"mse": mse, "r2": r2, "fit_time": fit_time, "pred_time": pred_time}

# ============================================================================
# Experiment 5: KV Cache Speedup
# ============================================================================
def test_kv_cache_speedup():
    print_section("Experiment 5: KV Cache Speedup")

    X, y = make_classification(
        n_samples=50100, n_features=100, n_informative=50, n_classes=10, random_state=42
    )
    X_train, X_test = X[:50000], X[50000:]
    y_train, y_test = y[:50000], y[50000:]

    print_result("Train samples", X_train.shape[0])
    print_result("Test samples", X_test.shape[0])
    print_result("Features", X_train.shape[1])

    # Without KV cache
    print("\n[Mode 1] fit_preprocessors (no KV cache)")
    clf_no_cache = TabPFNClassifier(
        fit_mode="fit_preprocessors",
        n_estimators=1,
        device=DEVICE,
        model_path=MODEL_PATH_CLASSIFIER
    )

    start = time.time()
    clf_no_cache.fit(X_train, y_train)
    fit_time_no_cache = time.time() - start

    # Multiple predictions to get average
    pred_times = []
    for _ in range(3):
        start = time.time()
        _ = clf_no_cache.predict(X_test)
        pred_times.append(time.time() - start)
    pred_time_no_cache = np.mean(pred_times)

    print_result("Fit time", f"{fit_time_no_cache:.3f}s")
    print_result("Predict time (avg)", f"{pred_time_no_cache:.3f}s")

    # With KV cache
    print("\n[Mode 2] fit_with_cache (KV cache enabled)")
    clf_cache = TabPFNClassifier(
        fit_mode="fit_with_cache",
        n_estimators=1,
        device=DEVICE,
        model_path=MODEL_PATH_CLASSIFIER
    )

    start = time.time()
    clf_cache.fit(X_train, y_train)
    fit_time_cache = time.time() - start

    pred_times = []
    for _ in range(3):
        start = time.time()
        _ = clf_cache.predict(X_test)
        pred_times.append(time.time() - start)
    pred_time_cache = np.mean(pred_times)

    print_result("Fit time", f"{fit_time_cache:.3f}s")
    print_result("Predict time (avg)", f"{pred_time_cache:.3f}s")

    speedup = pred_time_no_cache / pred_time_cache
    print(f"\n[Comparison]")
    print_result("Prediction speedup", f"{speedup:.1f}x")
    print_result("ms per test sample (no cache)", f"{pred_time_no_cache * 1000 / len(X_test):.2f}")
    print_result("ms per test sample (cached)", f"{pred_time_cache * 1000 / len(X_test):.2f}")

    return {
        "speedup": speedup,
        "fit_time_no_cache": fit_time_no_cache,
        "pred_time_no_cache": pred_time_no_cache,
        "fit_time_cache": fit_time_cache,
        "pred_time_cache": pred_time_cache
    }

# ============================================================================
# Experiment 6: Scalability Test
# ============================================================================
def test_scalability():
    print_section("Experiment 6: Scalability Test")

    sizes = [1000, 5000, 10000, 50000]
    results = []

    for n_samples in sizes:
        print(f"\n  Testing {n_samples:,} samples...")

        X, y = make_classification(
            n_samples=n_samples, n_features=50, n_informative=30,
            n_classes=5, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        clf = TabPFNClassifier(n_estimators=2, device=DEVICE, model_path=MODEL_PATH_CLASSIFIER)

        start = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - start

        start = time.time()
        y_pred = clf.predict(X_test)
        pred_time = time.time() - start

        acc = accuracy_score(y_test, y_pred)

        print(f"    Accuracy: {acc:.4f}, Fit: {fit_time:.2f}s, Predict: {pred_time:.2f}s")

        results.append({
            "n_samples": n_samples,
            "acc": acc,
            "fit_time": fit_time,
            "pred_time": pred_time,
            "total_time": fit_time + pred_time
        })

    return results

# ============================================================================
# Main
# ============================================================================
def main():
    print_header("TabPFN-3 Experiment Reproduction Suite")

    # System info
    print("\nSystem Information:")
    print_result("PyTorch version", torch.__version__)
    print_result("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print_result("GPU", torch.cuda.get_device_name(0))
        print_result("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_results = {}

    # Run experiments
    try:
        all_results["binary"] = test_binary_classification()
    except Exception as e:
        print(f"Error in binary classification: {e}")

    try:
        all_results["multiclass"] = test_multiclass_classification()
    except Exception as e:
        print(f"Error in multiclass classification: {e}")

    try:
        all_results["many_class"] = test_many_class_classification()
    except Exception as e:
        print(f"Error in many-class classification: {e}")

    try:
        all_results["regression"] = test_regression()
    except Exception as e:
        print(f"Error in regression: {e}")

    try:
        all_results["kv_cache"] = test_kv_cache_speedup()
    except Exception as e:
        print(f"Error in KV cache test: {e}")

    try:
        all_results["scalability"] = test_scalability()
    except Exception as e:
        print(f"Error in scalability test: {e}")

    # Summary
    print_header("Experiment Summary")
    print("\n✓ All experiments completed!")
    print("\nKey Findings:")
    if "binary" in all_results:
        print(f"  - Binary Classification Accuracy: {all_results['binary']['acc']:.4f}")
    if "many_class" in all_results:
        print(f"  - Many-Class (50 classes) Accuracy: {all_results['many_class']['acc']:.4f}")
    if "kv_cache" in all_results:
        print(f"  - KV Cache Speedup: {all_results['kv_cache']['speedup']:.1f}x")
    if "regression" in all_results:
        print(f"  - Regression R² Score: {all_results['regression']['r2']:.4f}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
