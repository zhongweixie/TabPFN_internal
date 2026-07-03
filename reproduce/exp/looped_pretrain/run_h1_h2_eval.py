#!/usr/bin/env python
"""H1 (confidence threshold) and H2 (KL stability) zero-retrain halting eval.

Both methods use the curric-60K model (which works at k=1..6) and apply
a per-row stopping rule post-hoc:

H1: stop at the first k where max_softmax_prob(test_row) >= threshold
H2: stop at the first k where KL(p_k || p_{k-1}) < threshold  (converged)

For each TabArena dataset:
  - Run curric-60K at k=1..6, collect all probas
  - Tune threshold via 3-fold CV on the training split (treating a subset of
    train rows as held-out val rows with the rest as context)
  - Apply tuned threshold to test rows
  - Report AUC vs fixed-k baselines

Usage:
  python run_h1_h2_eval.py --ckpt .../step-60000.ckpt --out_json h1h2_results.json
  python run_h1_h2_eval.py --method h1  # default both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from run_benchmark_eval import build_model
from run_tabarena_eval import load_task, score, TASKS_JSON, CTX_CAP, TEST_CAP

DEVICE = "cuda"
K_MIN = 1
K_MAX = 6
SEED = 42

# H1 candidate thresholds (max-softmax confidence to stop)
H1_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# H2 candidate thresholds (per-row mean KL to continue to next k)
H2_THRESHOLDS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

# Number of CV folds for threshold tuning on train split
N_CV_FOLDS = 3
# Max train rows used for CV (per fold's val set is CTX_CAP//3)
CV_CTX_CAP = CTX_CAP // 2   # context for CV val rows
CV_VAL_CAP = 200             # val rows evaluated per fold


# ─── multi-k inference ────────────────────────────────────────────────────────

@torch.no_grad()
def predict_all_k(model, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray) -> dict[int, np.ndarray]:
    """Run model at k=1..K_MAX, return {k: proba (M, C)} for all k values.

    We rebind loop_k on the model (per-instance attribute), so no global state
    contamination between k values."""
    import looped_step2 as L
    from run_benchmark_eval import predict_proba

    results: dict[int, np.ndarray] = {}
    for k in range(K_MIN, K_MAX + 1):
        L.set_loop_on_model(model, k)
        results[k] = predict_proba(model, X_train, y_train, X_test)
    return results


# ─── stopping policies ────────────────────────────────────────────────────────

def h1_select_proba(proba_by_k: dict[int, np.ndarray],
                    threshold: float) -> np.ndarray:
    """Per-row: use k=1 if confident enough, else k=2, ..., k=K_MAX.
    Returns (M, C) probability matrix."""
    M, C = proba_by_k[K_MIN].shape
    final = np.empty((M, C), dtype=np.float32)
    for row_i in range(M):
        chosen = proba_by_k[K_MAX][row_i]  # default: last k
        for k in range(K_MIN, K_MAX + 1):
            p = proba_by_k[k][row_i]
            if p.max() >= threshold:
                chosen = p
                break
        final[row_i] = chosen
    return final


def kl_row(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) for a single probability vector pair."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def h2_select_proba(proba_by_k: dict[int, np.ndarray],
                    threshold: float) -> np.ndarray:
    """Per-row: stop when KL(p_k || p_{k-1}) < threshold (converged).
    Start deciding from k=2 (need a previous step to compare against).
    Returns (M, C)."""
    M, C = proba_by_k[K_MIN].shape
    final = np.empty((M, C), dtype=np.float32)
    for row_i in range(M):
        chosen = proba_by_k[K_MAX][row_i]
        for k in range(K_MIN + 1, K_MAX + 1):
            kl = kl_row(proba_by_k[k][row_i], proba_by_k[k - 1][row_i])
            if kl < threshold:
                # converged at k: use k-1 (the one that didn't change much)
                chosen = proba_by_k[k - 1][row_i]
                break
        final[row_i] = chosen
    return final


# ─── threshold CV on training set ─────────────────────────────────────────────

def cv_tune_threshold(model, X_train: np.ndarray, y_train: np.ndarray,
                      method: str, thresholds: list[float]) -> float:
    """Tune threshold via CV on the training split.

    Splits train into CV_CTX_CAP context rows + CV_VAL_CAP val rows (N_CV_FOLDS folds).
    For each threshold, measures mean AUC over folds. Returns best threshold."""
    rng = np.random.RandomState(SEED)
    n_classes = int(y_train.max()) + 1
    fold_aucs: dict[float, list[float]] = {t: [] for t in thresholds}

    n_train = len(X_train)
    for fold in range(N_CV_FOLDS):
        # Shuffle once per fold
        perm = rng.permutation(n_train)
        # Take the first CV_VAL_CAP as val, rest (up to CV_CTX_CAP) as context
        n_val = min(CV_VAL_CAP, n_train // (N_CV_FOLDS + 1))
        if n_val < 10:
            continue  # too few samples for CV
        val_idx = perm[:n_val]
        ctx_idx = perm[n_val: n_val + CV_CTX_CAP]
        Xctx, yctx = X_train[ctx_idx], y_train[ctx_idx]
        Xval, yval = X_train[val_idx], y_train[val_idx]

        # Multi-k inference on val rows (using train-subset as context)
        proba_by_k = predict_all_k(model, Xctx, yctx, Xval)

        for t in thresholds:
            if method == "h1":
                p = h1_select_proba(proba_by_k, t)
            else:
                p = h2_select_proba(proba_by_k, t)
            _, auc = score(yval, p, n_classes)
            if not np.isnan(auc):
                fold_aucs[t].append(auc)

    # Pick threshold with best mean CV AUC (fallback: median of K_MAX)
    best_t = thresholds[len(thresholds) // 2]  # default: median threshold
    best_auc = -1.0
    for t in thresholds:
        aucs = fold_aucs[t]
        if aucs:
            m = np.mean(aucs)
            if m > best_auc:
                best_auc = m
                best_t = t
    return best_t


# ─── mean k usage ─────────────────────────────────────────────────────────────

def mean_k_used(proba_by_k: dict[int, np.ndarray],
                method: str, threshold: float) -> float:
    """Average k actually used per test row (measure of compute saved)."""
    M = proba_by_k[K_MIN].shape[0]
    total_k = 0
    for row_i in range(M):
        for k in range(K_MIN, K_MAX + 1):
            if method == "h1":
                stop = proba_by_k[k][row_i].max() >= threshold
            else:
                if k == K_MIN:
                    stop = False
                else:
                    stop = kl_row(proba_by_k[k][row_i],
                                  proba_by_k[k - 1][row_i]) < threshold
            if stop or k == K_MAX:
                total_k += k
                break
    return total_k / M


# ─── main evaluation loop ─────────────────────────────────────────────────────

def run_eval(ckpt_path: str, methods: list[str], out_json: str):
    import looped_step2 as L
    L.install_looped_forward()

    print(f"[H1/H2] Loading model from {ckpt_path}", flush=True)
    model = build_model(ckpt_path, nlayers=12, loop_k=1)  # loop_k overridden per-row

    tasks = json.load(open(TASKS_JSON))
    clf = [r for r in tasks if 2 <= r["cls"] <= 10]
    clf = sorted(clf, key=lambda r: r["rows"])
    print(f"[H1/H2] {len(clf)} TabArena classification datasets", flush=True)

    # Initialize results with all keys
    results: dict = {
        "config": {"ckpt": ckpt_path, "methods": methods,
                   "K_MIN": K_MIN, "K_MAX": K_MAX},
        "per_dataset": {},
    }
    for m in methods:
        results[f"{m}_aucs"] = []
        results[f"{m}_mean_k"] = []
    results["fixed_k_aucs"] = {k: [] for k in range(K_MIN, K_MAX + 1)}

    for r in clf:
        name, tid, ncls = r["name"], r["task"], r["cls"]
        t0 = time.time()
        try:
            Xtr, ytr, Xte, yte = load_task(tid)
        except Exception as e:
            print(f"  SKIP {name}: {e}", flush=True)
            continue
        ncls_eff = int(max(ytr.max(), yte.max())) + 1

        # Run model at all k values (test split)
        proba_by_k = predict_all_k(model, Xtr, ytr, Xte)

        # Fixed-k baselines
        fixed_k_auc: dict[int, float] = {}
        for k in range(K_MIN, K_MAX + 1):
            _, auc = score(yte, proba_by_k[k], ncls_eff)
            fixed_k_auc[k] = auc
            results["fixed_k_aucs"][k].append(auc)

        ds_result: dict = {"fixed_k": fixed_k_auc}
        line = f"  {name[:32]:32s}"
        for k in [1, 4, 6]:
            line += f" k{k}={fixed_k_auc[k]:.3f}"

        for method in methods:
            thresholds = H1_THRESHOLDS if method == "h1" else H2_THRESHOLDS

            # Tune threshold via CV on train split
            best_t = cv_tune_threshold(model, Xtr, ytr, method, thresholds)

            # Apply to test rows
            if method == "h1":
                p = h1_select_proba(proba_by_k, best_t)
            else:
                p = h2_select_proba(proba_by_k, best_t)

            _, auc = score(yte, p, ncls_eff)
            mk = mean_k_used(proba_by_k, method, best_t)
            results[f"{method}_aucs"].append(auc)
            results[f"{method}_mean_k"].append(mk)
            ds_result[method] = {"auc": auc, "threshold": best_t, "mean_k": mk}
            line += f" | {method.upper()}(t={best_t:.3f}) auc={auc:.3f} k={mk:.2f}"

        results["per_dataset"][name] = ds_result
        print(line + f"  ({time.time()-t0:.1f}s)", flush=True)

    # Summary
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for k in [1, 3, 4, 6]:
        aucs = [a for a in results["fixed_k_aucs"][k] if not np.isnan(a)]
        print(f"  fixed k={k:d}: AUC={np.mean(aucs):.4f}  (n={len(aucs)})")
    for method in methods:
        aucs = [a for a in results[f"{method}_aucs"] if not np.isnan(a)]
        mk = np.mean(results[f"{method}_mean_k"])
        print(f"  {method.upper():4s} adaptive: AUC={np.mean(aucs):.4f}"
              f"  mean_k={mk:.2f}  (n={len(aucs)})")

    json.dump(results, open(out_json, "w"), indent=2)
    print(f"\n[H1/H2] Results saved to {out_json}", flush=True)


def main():
    p = argparse.ArgumentParser(description="H1/H2 zero-retrain halting eval")
    p.add_argument("--ckpt", default=os.path.join(
        REPRO, "ckpt", "curric_k1to6_12L_80000", "step-60000.ckpt"),
        help="Checkpoint path (default: curric 60K)")
    p.add_argument("--methods", default="h1,h2",
        help="Comma-separated methods to run (h1, h2, or h1,h2)")
    p.add_argument("--out_json", default=os.path.join(REPRO, "h1h2_results.json"))
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run_eval(args.ckpt, methods, args.out_json)


if __name__ == "__main__":
    main()
