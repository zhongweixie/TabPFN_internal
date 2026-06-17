#!/usr/bin/env python3
"""
TabPFN-3 大数据扩展性验证 (Section 3.2.1)
论文宣称: 支持 1M 行数据
当前已验证: 100K
目标: 推进到 250K -> 500K -> 1M, 诚实报告 L20 (48GB) 实际上限

关键参数:
  ignore_pretraining_limits=True  # 突破10K行预训练软限制
  memory_saving_mode=True         # row-chunking 降低内存峰值
  n_estimators=2                  # 大数据下减少集成数
  fit_mode="fit_with_cache"       # KV cache (复用已验证基础设施)
"""

import os
os.environ["TABPFN_NO_BROWSER"] = "0"

import gc
import time
import numpy as np
import torch
import psutil
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

MODEL_PATH = "/home/zxiebk/workspace/model/tabpfn_3/tabpfn-v3-classifier-v3_default.ckpt"
DEVICE = "cuda"

# 测试规模: 从已验证的100K推进
SIZES = [100_000, 250_000, 500_000, 1_000_000]
N_FEATURES = 100
N_CLASSES = 5
TEST_SIZE = 5_000  # 固定测试集大小, 隔离训练规模的影响


def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def gpu_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def cpu_mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


print("=" * 80)
print("TabPFN-3 大数据扩展性验证 (Section 3.2.1)")
print("=" * 80)
print(f"\n论文宣称: 支持 1M 行")
print(f"硬件: L20 (48GB)")
print(f"测试规模: {[f'{s:,}' for s in SIZES]}")
print(f"特征数: {N_FEATURES}, 类别数: {N_CLASSES}, 固定测试集: {TEST_SIZE:,}")

results = []

for n_samples in SIZES:
    print("\n" + "=" * 80)
    print(f"测试规模: {n_samples:,} 行")
    print("=" * 80)

    record = {"n_samples": n_samples}
    try:
        print(f"  生成数据 ({n_samples:,} x {N_FEATURES})...")
        t0 = time.time()
        X, y = make_classification(
            n_samples=n_samples + TEST_SIZE,
            n_features=N_FEATURES,
            n_informative=70,
            n_classes=N_CLASSES,
            random_state=42,
        )
        X_train, X_test = X[:n_samples], X[n_samples:]
        y_train, y_test = y[:n_samples], y[n_samples:]
        print(f"    数据生成: {time.time() - t0:.1f}s | 训练 {X_train.shape[0]:,} 测试 {X_test.shape[0]:,}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        clf = TabPFNClassifier(
            n_estimators=2,
            device=DEVICE,
            model_path=MODEL_PATH,
            ignore_pretraining_limits=True,
            memory_saving_mode=True,
            fit_mode="fit_with_cache",
        )

        print(f"  开始 fit...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        fit_peak = gpu_peak_mb()
        print(f"    fit 时间: {fit_time:.2f}s | GPU峰值: {fit_peak:.0f} MB")

        print(f"  开始 predict ({TEST_SIZE:,} 行)...")
        t0 = time.time()
        proba = clf.predict_proba(X_test)
        pred_time = time.time() - t0
        y_pred = proba.argmax(axis=1)
        pred_peak = gpu_peak_mb()

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, proba, multi_class="ovr")
        except Exception:
            auc = float("nan")

        print(f"    predict 时间: {pred_time:.2f}s | GPU峰值: {pred_peak:.0f} MB")
        print(f"    准确率: {acc:.4f} | AUC(ovr): {auc:.4f}")

        record.update({
            "success": True,
            "fit_time": fit_time,
            "pred_time": pred_time,
            "accuracy": acc,
            "auc": auc,
            "gpu_peak_mb": max(fit_peak, pred_peak),
            "cpu_mem_mb": cpu_mem_mb(),
        })
        print(f"  ✓ {n_samples:,} 行测试成功!")

        del clf, X, y, X_train, X_test, y_train, y_test, proba
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        err = str(e)
        print(f"  ✗ {n_samples:,} 行失败: {err[:200]}")
        is_oom = "out of memory" in err.lower() or "CUDA" in err
        record.update({"success": False, "error": err[:300], "oom": is_oom})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        traceback.print_exc()

    results.append(record)

# ============================================================================
# 汇总
# ============================================================================
print("\n" + "=" * 80)
print("汇总: 大数据扩展性结果")
print("=" * 80)
print(f"\n{'规模':>12} | {'状态':>6} | {'fit(s)':>8} | {'pred(s)':>8} | {'准确率':>8} | {'AUC':>7} | {'GPU峰值(MB)':>12}")
print("-" * 80)
for r in results:
    if r.get("success"):
        print(f"{r['n_samples']:>12,} | {'✓':>6} | {r['fit_time']:>8.2f} | "
              f"{r['pred_time']:>8.2f} | {r['accuracy']:>8.4f} | {r['auc']:>7.4f} | {r['gpu_peak_mb']:>12.0f}")
    else:
        tag = "OOM" if r.get("oom") else "FAIL"
        print(f"{r['n_samples']:>12,} | {tag:>6} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>7} | {'—':>12}")

succeeded = [r for r in results if r.get("success")]
if succeeded:
    max_n = max(r["n_samples"] for r in succeeded)
    print(f"\n实际验证上限 (L20 48GB): {max_n:,} 行")
    paper_claim = 1_000_000
    print(f"论文宣称: {paper_claim:,} 行")
    print(f"达成比例: {max_n / paper_claim * 100:.1f}% (论文宣称应在更大显存上验证, 如 H100/A100 80GB)")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)
