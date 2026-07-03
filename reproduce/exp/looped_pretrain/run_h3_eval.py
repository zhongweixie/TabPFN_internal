#!/usr/bin/env python
"""Evaluate H3 halt MLP checkpoint on TabArena.

Loads the fine-tuned halt_head + transformer, then for each TabArena dataset:
  - Runs the model at k=1..6 collecting per-step hidden-state halt signals
  - Uses the halt_head to decide when to stop (halt_prob > threshold)
  - Also reports fixed-k baselines for comparison

Usage:
  python run_h3_eval.py --ckpt .../ckpt/h3_halt_mlp_10k/step-10000.ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from run_benchmark_eval import predict_proba
from run_tabarena_eval import load_task, score, TASKS_JSON

DEVICE = "cuda"
K_MIN, K_MAX = 1, 6
SEED = 42
HALT_THRESHOLD = float(os.environ.get("HALT_THRESHOLD", "0.5"))


def build_h3_model(ckpt_path: str, nlayers: int = 12, loop_k: int = 6):
    """Load H3-finetuned model (strict=False to tolerate halt_head keys)."""
    from taco.model.taco_model import TACO
    from taco.model.tabpfn_arch.model.loading import ModelConfig
    import looped_step2 as L

    cfg = ModelConfig(
        emsize=192, nhead=6, nlayers=nlayers, nhid_factor=4,
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
    # Load with strict=False: halt_head keys are extra, that's fine
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[H3-eval] loaded {ckpt_path}: missing={len(missing)} extra={len(unexpected)}")

    # Rebuild halt_head using the same HaltHead class from training
    from train_h3_halt_mlp import HaltHead
    emsize = 192
    halt_head = HaltHead(emsize=emsize, hidden=64).to(DEVICE)
    # saved keys: "halt_head.net.0.weight" → strip "halt_head." → "net.0.weight" ✓
    hh_state = {k[len("halt_head."):]: v for k, v in state.items()
                if k.startswith("halt_head.")}
    if hh_state:
        halt_head.load_state_dict(hh_state, strict=True)
        print(f"[H3-eval] halt_head loaded ({len(hh_state)} tensors)")
    else:
        print("[H3-eval] WARNING: no halt_head weights in checkpoint, using random init")

    model.eval()
    L.install_looped_forward()
    L.set_loop_on_model(model, loop_k)
    return model, halt_head


@torch.no_grad()
def predict_h3(model, halt_head, X_train, y_train, X_test,
               threshold: float = 0.5):
    """Run inference with H3 halt head guiding k selection per dataset.
    Returns (M, C) probabilities and mean k used."""
    import looped_step2 as L

    n_train = len(X_train)
    B = 1  # batch size = 1 for eval

    X_all = np.concatenate([X_train, X_test], axis=0)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    n_classes = int(y_train.max()) + 1

    # Install halt-capture forward from train_h3_halt_mlp
    # (not imported here to keep this script standalone — replicate logic inline)
    intermediates = []

    # Capture hidden states by hooking into the looped forward
    from taco.model.tabpfn_arch.model.transformer import LayerStack
    orig_fwd = LayerStack.forward

    def hooked_fwd(self, x, *, half_layers=False, **kwargs):
        k = getattr(self, "_loop_k", L.LOOP_K)
        alpha = getattr(self, "_reinject_alpha", L.REINJECT_ALPHA)
        h0 = x
        out = x
        Bx = x.shape[0]
        for i in range(k):
            out = L._ORIG_LAYERSTACK_FWD(self, out, half_layers=half_layers, **kwargs)
            if n_train > 0 and out.shape[1] > n_train:
                h_test = out[:, n_train:]
                h_mean = h_test.reshape(Bx, -1, h_test.shape[-1]).mean(dim=1)
                intermediates.append((i, h_mean))
            if i < k - 1:
                out = out + alpha * h0
        return out

    # Run at each k, check halt signal
    final_proba = None
    k_used = K_MAX
    for k in range(K_MIN, K_MAX + 1):
        intermediates.clear()
        L.set_loop_on_model(model, k)
        LayerStack.forward = hooked_fwd
        try:
            logits = model(X_t, y_t)
        finally:
            LayerStack.forward = orig_fwd

        proba = torch.softmax(logits[0], dim=-1).cpu().numpy()
        if final_proba is None:
            final_proba = proba

        # Get halt signal from last captured intermediate
        if intermediates:
            _, h_mean = intermediates[-1]
            halt_logit = halt_head(h_mean)  # (1, 1)
            halt_prob = torch.sigmoid(halt_logit).mean().item()
            if halt_prob >= threshold or k == K_MAX:
                final_proba = proba
                k_used = k
                break
        else:
            final_proba = proba
            k_used = k
            break

    if final_proba.shape[1] < n_classes:
        pad = np.zeros((len(X_test), n_classes - final_proba.shape[1]))
        final_proba = np.concatenate([final_proba, pad], axis=1)
    return final_proba[:, :n_classes], k_used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(
        REPRO, "ckpt", "h3_halt_mlp_10k", "step-10000.ckpt"))
    p.add_argument("--threshold", type=float, default=HALT_THRESHOLD)
    p.add_argument("--out_json", default=os.path.join(REPRO, "h3_eval_results.json"))
    args = p.parse_args()

    import looped_step2 as L
    L.install_looped_forward()

    print(f"[H3-eval] Loading model from {args.ckpt}", flush=True)
    model, halt_head = build_h3_model(args.ckpt, nlayers=12, loop_k=K_MAX)

    tasks = json.load(open(TASKS_JSON))
    clf = [r for r in tasks if 2 <= r["cls"] <= 10]
    clf = sorted(clf, key=lambda r: r["rows"])
    print(f"[H3-eval] {len(clf)} TabArena clf datasets, threshold={args.threshold}",
          flush=True)

    results = {"h3_aucs": [], "h3_k": [], "fixed_k_aucs": {k: [] for k in range(1, 7)},
               "per_dataset": {}}

    for r in clf:
        name, tid, ncls = r["name"], r["task"], r["cls"]
        t0 = time.time()
        try:
            Xtr, ytr, Xte, yte = load_task(tid)
        except Exception as e:
            print(f"  SKIP {name}: {e}", flush=True); continue
        ncls_eff = int(max(ytr.max(), yte.max())) + 1

        # Fixed-k baselines
        fixed_auc = {}
        for k in range(K_MIN, K_MAX + 1):
            L.set_loop_on_model(model, k)
            p = predict_proba(model, Xtr, ytr, Xte)
            _, auc = score(yte, p, ncls_eff)
            fixed_auc[k] = auc
            results["fixed_k_aucs"][k].append(auc)

        # H3 halt-guided
        p_h3, k_used = predict_h3(model, halt_head, Xtr, ytr, Xte,
                                   threshold=args.threshold)
        _, auc_h3 = score(yte, p_h3, ncls_eff)
        results["h3_aucs"].append(auc_h3)
        results["h3_k"].append(k_used)
        results["per_dataset"][name] = {"fixed": fixed_auc, "h3": auc_h3, "k_used": k_used}

        line = (f"  {name[:32]:32s}"
                f" k4={fixed_auc[4]:.3f} k6={fixed_auc[6]:.3f}"
                f" | H3(k={k_used}) auc={auc_h3:.3f}  ({time.time()-t0:.1f}s)")
        print(line, flush=True)

    print(f"\n{'='*65}\nSUMMARY\n{'='*65}")
    for k in [1, 3, 4, 6]:
        aucs = [a for a in results["fixed_k_aucs"][k] if not np.isnan(a)]
        print(f"  fixed k={k}: AUC={np.mean(aucs):.4f}  (n={len(aucs)})")
    h3a = [a for a in results["h3_aucs"] if not np.isnan(a)]
    print(f"  H3 halt: AUC={np.mean(h3a):.4f}  mean_k={np.mean(results['h3_k']):.2f}")

    json.dump(results, open(args.out_json, "w"), indent=2)
    print(f"\n[H3-eval] Saved to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
