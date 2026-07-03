#!/usr/bin/env python
"""H6 budget-conditional TabArena eval.

Four comparisons:
  H6-k4    : H6 model at fixed k=4 with budget_emb[4] (vs loopk4: does multi-k training help?)
  H6-adapt  : H6 + H1-style confidence threshold to select k adaptively
  H6-oracle : H6 best k per dataset (theoretical ceiling with H6 representations)
  ref-loopk4: original loopk4 checkpoint (external baseline)

IMPORTANT: every predict call must be wrapped with enable_budget(k)/disable_budget().

Usage:
  CUDA_VISIBLE_DEVICES=5 python run_h6_eval.py \
      --ckpt .../ckpt/h6_budget_cond_12L_80000/step-80000.ckpt
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from run_benchmark_eval import build_model, predict_proba
from run_tabarena_eval import load_task, score, TASKS_JSON
import looped_step2 as L

DEVICE = "cuda"
K_MIN, K_MAX = 1, 6
SEED = 42


def build_h6_model(ckpt_path: str):
    """Load H6 checkpoint (has extra budget_embedding keys → strict=False)."""
    from taco.model.taco_model import TACO
    from taco.model.tabpfn_arch.model.loading import ModelConfig
    from train_h6_budget_cond import BudgetEmbedding, install_budget_forward, _BUDGET_STATE

    cfg = ModelConfig(
        emsize=192, nhead=6, nlayers=12, nhid_factor=4,
        features_per_group=2, max_num_classes=10, max_num_features=85,
        num_buckets=1000, dropout=0.0, encoder_use_bias=False,
        feature_positional_embedding="subspace", multiquery_item_attention=False,
        nan_handling_enabled=True, nan_handling_y_encoder=True,
        normalize_by_used_features=True, normalize_on_train_only=True,
        normalize_to_ranking=False, normalize_x=True,
        recompute_attn=False, recompute_layer=True,
        remove_empty_features=True, remove_outliers=False,
        remove_duplicate_features=False, two_sets_of_queries=False,
        use_separate_decoder=False, use_flash_attention=True,
        multiquery_item_attention_for_test_set=True,
        attention_init_gain=1.0, dag_pos_enc_dim=None,
        item_attention_type="full", feature_attention_type="full", seed=0,
    )
    model = TACO(use_compressor=False, row_compression_percentage=0.5,
                 rcp_sampling="none", new_tabpfn_config=cfg).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[H6-eval] loaded {ckpt_path}: missing={len(missing)} extra={len(unexpected)}")

    budget_emb = BudgetEmbedding(k_max=K_MAX, emsize=192).to(DEVICE)
    be_state = {k[len("budget_embedding."):]: v for k, v in state.items()
                if k.startswith("budget_embedding.")}
    if be_state:
        budget_emb.load_state_dict(be_state, strict=True)
        print(f"[H6-eval] budget_embedding loaded ({len(be_state)} tensors)")
    else:
        print("[H6-eval] WARNING: no budget_embedding in checkpoint")

    model.budget_embedding = budget_emb
    _BUDGET_STATE["budget_emb"] = budget_emb

    model.eval()
    install_budget_forward()
    return model


@torch.no_grad()
def predict_h6_at_k(model, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, k: int) -> np.ndarray:
    """Run H6 at a specific k with correct budget conditioning.
    MUST use enable_budget(k) / disable_budget() — fallback is unconditioned."""
    from train_h6_budget_cond import enable_budget, disable_budget

    n_classes = int(y_train.max()) + 1
    M = len(X_test)
    X_all = np.concatenate([X_train, X_test], axis=0)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    L.set_loop_on_model(model, k)
    enable_budget(k=k)
    logits = model(X_t, y_t)
    disable_budget()

    proba = torch.softmax(logits[0], dim=-1).cpu().numpy()
    if proba.shape[1] < n_classes:
        proba = np.concatenate([proba, np.zeros((M, n_classes - proba.shape[1]))], 1)
    return proba[:, :n_classes]


def adaptive_k_h1(proba_by_k: dict, threshold: float) -> np.ndarray:
    """H1-style: per-row stop when max confidence >= threshold."""
    M, C = proba_by_k[K_MIN].shape
    final = np.empty((M, C), dtype=np.float32)
    for row_i in range(M):
        chosen = proba_by_k[K_MAX][row_i]
        for k in range(K_MIN, K_MAX + 1):
            if proba_by_k[k][row_i].max() >= threshold:
                chosen = proba_by_k[k][row_i]
                break
        final[row_i] = chosen
    return final


def cv_tune_threshold(model, X_train, y_train, thresholds, n_folds=3):
    """Tune H1 threshold via CV on training split."""
    rng = np.random.RandomState(SEED)
    n = len(X_train)
    n_val = min(200, n // (n_folds + 1))
    n_ctx = min(500, n)
    fold_aucs = {t: [] for t in thresholds}
    n_classes = int(y_train.max()) + 1

    for _ in range(n_folds):
        perm = rng.permutation(n)
        val_idx, ctx_idx = perm[:n_val], perm[n_val:n_val + n_ctx]
        Xctx, yctx = X_train[ctx_idx], y_train[ctx_idx]
        Xval, yval = X_train[val_idx], y_train[val_idx]

        pby_k = {k: predict_h6_at_k(model, Xctx, yctx, Xval, k)
                 for k in range(K_MIN, K_MAX + 1)}
        for t in thresholds:
            p = adaptive_k_h1(pby_k, t)
            _, auc = score(yval, p, n_classes)
            if not np.isnan(auc):
                fold_aucs[t].append(auc)

    best_t, best_auc = thresholds[len(thresholds) // 2], -1.0
    for t in thresholds:
        if fold_aucs[t]:
            m = np.mean(fold_aucs[t])
            if m > best_auc:
                best_auc, best_t = m, t
    return best_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ref_ckpt", default=os.path.join(
        REPRO, "ckpt", "loopk4_12L_80000", "step-80000.ckpt"))
    p.add_argument("--out_json", default=os.path.join(REPRO, "h6_eval_results.json"))
    args = p.parse_args()

    L.install_looped_forward()
    model = build_h6_model(args.ckpt)

    ref_model = None
    if os.path.exists(args.ref_ckpt):
        ref_model = build_model(args.ref_ckpt, nlayers=12, loop_k=4)
        print("[H6-eval] ref loopk4 loaded", flush=True)

    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

    tasks = json.load(open(TASKS_JSON))
    clf = sorted([r for r in tasks if 2 <= r["cls"] <= 10], key=lambda r: r["rows"])
    print(f"[H6-eval] {len(clf)} datasets", flush=True)

    results = {"h6_k4_aucs": [], "h6_adapt_aucs": [], "h6_oracle_aucs": [],
               "ref_k4_aucs": [], "h6_adapt_mean_k": [], "per_dataset": {}}

    for r in clf:
        name, tid = r["name"], r["task"]
        t0 = time.time()
        try:
            Xtr, ytr, Xte, yte = load_task(tid)
        except Exception as e:
            print(f"  SKIP {name}: {e}", flush=True); continue
        ncls_eff = int(max(ytr.max(), yte.max())) + 1

        # Run at all k values
        pby_k = {k: predict_h6_at_k(model, Xtr, ytr, Xte, k)
                 for k in range(K_MIN, K_MAX + 1)}

        # H6 at fixed k=4
        _, auc_k4 = score(yte, pby_k[4], ncls_eff)

        # H6 oracle (best k per dataset, not per row — simple dataset-level oracle)
        aucs_per_k = {k: score(yte, pby_k[k], ncls_eff)[1] for k in range(K_MIN, K_MAX + 1)}
        auc_oracle = max(v for v in aucs_per_k.values() if not np.isnan(v))
        best_k = max(aucs_per_k, key=lambda k: aucs_per_k[k] if not np.isnan(aucs_per_k[k]) else -1)

        # H6 adaptive (H1 threshold tuned via CV)
        best_t = cv_tune_threshold(model, Xtr, ytr, thresholds)
        p_adapt = adaptive_k_h1(pby_k, best_t)
        _, auc_adapt = score(yte, p_adapt, ncls_eff)
        mk_adapt = float(np.mean([
            next(k for k in range(K_MIN, K_MAX + 1) if pby_k[k][row_i].max() >= best_t or k == K_MAX)
            for row_i in range(len(Xte))
        ]))

        # Reference loopk4
        auc_ref = float("nan")
        if ref_model is not None:
            p_ref = predict_proba(ref_model, Xtr, ytr, Xte)
            _, auc_ref = score(yte, p_ref, ncls_eff)

        results["h6_k4_aucs"].append(auc_k4)
        results["h6_adapt_aucs"].append(auc_adapt)
        results["h6_oracle_aucs"].append(auc_oracle)
        results["ref_k4_aucs"].append(auc_ref)
        results["h6_adapt_mean_k"].append(mk_adapt)
        results["per_dataset"][name] = {
            "h6_k4": auc_k4, "h6_adapt": auc_adapt, "h6_oracle": auc_oracle,
            "ref_k4": auc_ref, "adapt_thresh": best_t, "oracle_best_k": best_k,
            "aucs_per_k": aucs_per_k,
        }

        ref_str = f" ref={auc_ref:.3f}" if not np.isnan(auc_ref) else ""
        print(f"  {name[:30]:30s} k4={auc_k4:.3f} adapt(t={best_t:.2f})={auc_adapt:.3f}"
              f" oracle(k={best_k})={auc_oracle:.3f}{ref_str}  ({time.time()-t0:.1f}s)",
              flush=True)

    print(f"\n{'='*65}\nSUMMARY\n{'='*65}")
    for tag, key in [("H6 fixed k=4  ", "h6_k4_aucs"),
                     ("H6 adaptive   ", "h6_adapt_aucs"),
                     ("H6 oracle     ", "h6_oracle_aucs"),
                     ("loopk4 ref    ", "ref_k4_aucs")]:
        aucs = [a for a in results[key] if not np.isnan(a)]
        if aucs:
            print(f"  {tag}: AUC={np.mean(aucs):.4f}  (n={len(aucs)})")
    print(f"  H6 adaptive mean_k: {np.mean(results['h6_adapt_mean_k']):.2f}")

    json.dump(results, open(args.out_json, "w"), indent=2)
    print(f"\n[H6-eval] Saved → {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
