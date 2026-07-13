#!/usr/bin/env python
"""H5 ACT TabArena eval.

Three comparisons:
  H5-ACT      : H5 model with ACT enabled (soft-weighted output, adaptive k)
  H5-base-k4  : H5 backbone at fixed k=4, ACT disabled (tests if fine-tuning hurt backbone)
  ref-loopk4  : original loopk4 checkpoint (external baseline, AUC≈0.817)

Usage:
  CUDA_VISIBLE_DEVICES=4 python run_h5_eval.py \
      --ckpt .../ckpt/h5_act_l001_12L_80000/step-80000.ckpt
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
_taco_src = os.environ.get("TACO_SRC", "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, _taco_src)
sys.path.insert(0, HERE)

from run_benchmark_eval import build_model, predict_proba
from run_tabarena_eval import load_task, score, TASKS_JSON
import looped_step2 as L

DEVICE = "cuda"
K_MAX = 6


def build_h5_model(ckpt_path: str):
    """Load H5 checkpoint (has extra act_halting_unit keys → strict=False)."""
    from taco.model.taco_model import TACO
    from taco.model.tabpfn_arch.model.loading import ModelConfig
    from train_h5_act import ACTHaltingUnit, install_act_forward, _ACT_STATE

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
    print(f"[H5-eval] loaded {ckpt_path}: missing={len(missing)} extra={len(unexpected)}")

    halt_unit = ACTHaltingUnit(emsize=192).to(DEVICE)
    hu_state = {k[len("act_halting_unit."):]: v for k, v in state.items()
                if k.startswith("act_halting_unit.")}
    if hu_state:
        halt_unit.load_state_dict(hu_state, strict=True)
        print(f"[H5-eval] halt_unit loaded ({len(hu_state)} tensors)")
    else:
        print("[H5-eval] WARNING: no act_halting_unit in checkpoint")

    model.act_halting_unit = halt_unit
    _ACT_STATE["halting_unit"] = halt_unit
    _ACT_STATE["k_max"] = K_MAX

    model.eval()
    install_act_forward()
    L.set_loop_on_model(model, K_MAX)
    return model


@torch.no_grad()
def predict_act(model, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray) -> tuple[np.ndarray, float]:
    """Run ACT-enabled inference. Returns (proba (M,C), mean_k)."""
    from train_h5_act import enable_act, disable_act, _ACT_STATE

    n_train = len(X_train)
    n_classes = int(y_train.max()) + 1
    M = len(X_test)

    X_all = np.concatenate([X_train, X_test], axis=0)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    enable_act(n_train=n_train)
    logits = model(X_t, y_t)       # ACT soft-weighted forward runs here
    mean_k = float(_ACT_STATE["ponder_cost"].item()) if _ACT_STATE["ponder_cost"] is not None else K_MAX
    disable_act()

    proba = torch.softmax(logits[0], dim=-1).cpu().numpy()
    if proba.shape[1] < n_classes:
        proba = np.concatenate([proba, np.zeros((M, n_classes - proba.shape[1]))], 1)
    return proba[:, :n_classes], mean_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="H5 ACT checkpoint path")
    p.add_argument("--ref_ckpt", default=os.path.join(
        REPRO, "ckpt", "loopk4_12L_80000", "step-80000.ckpt"))
    p.add_argument("--out_json", default=os.path.join(REPRO, "h5_eval_results.json"))
    args = p.parse_args()

    L.install_looped_forward()
    model = build_h5_model(args.ckpt)

    ref_model = None
    if os.path.exists(args.ref_ckpt):
        ref_model = build_model(args.ref_ckpt, nlayers=12, loop_k=4)
        # build_model() reinstalls the global fixed-loop LayerStack.forward.
        # Restore the ACT wrapper after constructing the reference, so
        # predict_act() below truly evaluates the H5 adaptive path.
        from train_h5_act import install_act_forward
        install_act_forward()
        print(f"[H5-eval] ref loopk4 loaded", flush=True)

    tasks = json.load(open(TASKS_JSON))
    clf = sorted([r for r in tasks if 2 <= r["cls"] <= 10], key=lambda r: r["rows"])
    print(f"[H5-eval] {len(clf)} datasets", flush=True)

    results = {"act_aucs": [], "base_k4_aucs": [], "ref_k4_aucs": [],
               "act_mean_k": [], "per_dataset": {}}

    for r in clf:
        name, tid = r["name"], r["task"]
        t0 = time.time()
        try:
            Xtr, ytr, Xte, yte = load_task(tid)
        except Exception as e:
            print(f"  SKIP {name}: {e}", flush=True); continue
        ncls_eff = int(max(ytr.max(), yte.max())) + 1

        p_act, mk = predict_act(model, Xtr, ytr, Xte)
        _, auc_act = score(yte, p_act, ncls_eff)

        from train_h5_act import disable_act
        disable_act()
        L.set_loop_on_model(model, 4)
        p_k4 = predict_proba(model, Xtr, ytr, Xte)
        _, auc_k4 = score(yte, p_k4, ncls_eff)

        auc_ref = float("nan")
        if ref_model is not None:
            p_ref = predict_proba(ref_model, Xtr, ytr, Xte)
            _, auc_ref = score(yte, p_ref, ncls_eff)

        results["act_aucs"].append(auc_act)
        results["base_k4_aucs"].append(auc_k4)
        results["ref_k4_aucs"].append(auc_ref)
        results["act_mean_k"].append(mk)
        results["per_dataset"][name] = {"act": auc_act, "base_k4": auc_k4,
                                         "ref_k4": auc_ref, "mean_k": mk}

        ref_str = f" ref={auc_ref:.3f}" if not np.isnan(auc_ref) else ""
        print(f"  {name[:32]:32s} k4={auc_k4:.3f} act={auc_act:.3f}{ref_str}"
              f" mk={mk:.1f}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n{'='*65}\nSUMMARY\n{'='*65}")
    for tag, key in [("H5 backbone k=4 ", "base_k4_aucs"),
                     ("H5 ACT adaptive", "act_aucs"),
                     ("loopk4 reference", "ref_k4_aucs")]:
        aucs = [a for a in results[key] if not np.isnan(a)]
        if aucs:
            print(f"  {tag}: AUC={np.mean(aucs):.4f}  (n={len(aucs)})")
    print(f"  mean k used (ACT): {np.mean(results['act_mean_k']):.2f}")

    json.dump(results, open(args.out_json, "w"), indent=2)
    print(f"\n[H5-eval] Saved → {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
