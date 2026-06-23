#!/usr/bin/env python
"""Benchmark eval: load trained checkpoints (c1/c2/c3) and evaluate on real
OpenML classification datasets. Reports accuracy + ROC-AUC per dataset + mean.

This bypasses TACOClassifier's auto-download and directly loads our custom
checkpoints into the TACO model, with the loop patch active for c2."""

from __future__ import annotations

import sys
import os
import numpy as np
import torch
from pathlib import Path
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine, load_iris
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, "/data/zxiebk/workspace/train/PFN/TabPFN/reproduce")

DEVICE = "cuda"
REPO = Path("/data/zxiebk/workspace/train/PFN/TabPFN/reproduce")
SEED = 42
SEEDS = [42, 123, 7]  # 3 splits per dataset for robustness
MAX_TRAIN = 1000   # cap train size for fair comparison (models trained on ≤600 rows)
MAX_TEST = 300

# Datasets: expanded to ~25, mix of easy/hard, binary/multiclass, small/medium
# Matches TACO paper spirit (TabArena-style classification datasets)
DATASETS = [
    # sklearn built-ins
    ("breast_cancer", None),
    ("iris", None),
    ("wine", None),
    # OpenML binary
    ("phoneme", "phoneme"),
    ("electricity", "electricity"),
    ("qsar-biodeg", "qsar-biodeg"),
    ("blood-transfusion", "blood-transfusion-service-center"),
    ("credit-g", "credit-g"),
    ("diabetes", "diabetes"),
    ("banknote", "banknote-authentication"),
    ("ilpd", "ilpd"),
    ("steel-plates", "steel-plates-fault"),
    ("ozone-1h", "ozone-level-8hr"),
    ("kc1", "kc1"),
    ("pc1", "pc1"),
    # OpenML multiclass
    ("vehicle", "vehicle"),
    ("segment", "segment"),
    ("satimage", "satimage"),
    ("vowel", "vowel"),
    ("JapaneseVowels", "JapaneseVowels"),
    ("har", "har"),
    ("letter", "letter"),
    ("pendigits", "pendigits"),
    ("optdigits", "optdigits"),
    ("mfeat-factors", "mfeat-factors"),
]


def load_dataset(name, openml_name):
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif name == "iris":
        X, y = load_iris(return_X_y=True)
    elif name == "wine":
        X, y = load_wine(return_X_y=True)
    else:
        d = fetch_openml(openml_name, version=1, as_frame=False, parser="liac-arff")
        X = d.data.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(d.target)
    # remove NaN rows
    mask = np.isfinite(X).all(1)
    X, y = X[mask], y[mask]
    return X.astype(np.float32), y.astype(np.int64)


def build_model(ckpt_path, nlayers, loop_k):
    """Load a TACO predictor-only model from checkpoint.

    IMPORTANT: always install the loop patch and set L.LOOP_K explicitly, even for
    loop_k=1. A prior bug left L.LOOP_K stale at 2 (set by a c2 load) so a subsequent
    c3 (loop_k=1) silently ran with 2 iterations — corrupting the c3 benchmark. Setting
    it every call (and LOOP_K=1 == byte-identical baseline, verified by forward-selftest)
    makes config loading order-independent."""
    from taco.model.taco_model import TACO
    from taco.model.tabpfn_arch.model.loading import ModelConfig

    os.environ["LOOP_K"] = str(loop_k)
    os.environ["REINJECT_ALPHA"] = "0.1"
    import looped_step2 as L
    L.LOOP_K = loop_k
    L.REINJECT_ALPHA = 0.1
    L.install_looped_forward()   # idempotent; LOOP_K=1 is the exact original forward

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
    model = TACO(
        use_compressor=False,
        row_compression_percentage=0.5,
        rcp_sampling="none",
        new_tabpfn_config=cfg,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, X_train, y_train, X_test):
    """Run inference: concatenate train+test, pass through model, get test logits."""
    n_classes = int(y_train.max()) + 1
    N = len(X_train)
    M = len(X_test)

    # Concatenate train + test
    X_all = np.concatenate([X_train, X_test], axis=0)  # (N+M, F)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, N+M, F)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, N)

    logits = model(X_t, y_t)  # (1, M, C)
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()  # (M, C)

    # Pad to n_classes if model outputs fewer
    if probs.shape[1] < n_classes:
        pad = np.zeros((M, n_classes - probs.shape[1]))
        probs = np.concatenate([probs, pad], axis=1)
    return probs[:, :n_classes]


def evaluate_all(model, tag):
    print(f"\n{'='*60}\n{tag}\n{'='*60}")
    results = []
    for name, openml_name in DATASETS:
        try:
            X, y = load_dataset(name, openml_name)
        except Exception as e:
            print(f"  {name}: SKIP ({e})")
            continue

        seed_accs, seed_aucs = [], []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            if len(Xtr) > MAX_TRAIN:
                idx = rng.choice(len(Xtr), MAX_TRAIN, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]
            if len(Xte) > MAX_TEST:
                idx = rng.choice(len(Xte), MAX_TEST, replace=False)
                Xte, yte = Xte[idx], yte[idx]

            probs = predict_proba(model, Xtr, ytr, Xte)
            pred = probs.argmax(1)
            acc = accuracy_score(yte, pred)
            try:
                if len(np.unique(yte)) == 2:
                    auc = roc_auc_score(yte, probs[:, 1])
                else:
                    auc = roc_auc_score(yte, probs, multi_class="ovr", average="macro",
                                        labels=list(range(probs.shape[1])))
            except ValueError:
                auc = float("nan")
            seed_accs.append(acc)
            seed_aucs.append(auc)

        mean_acc = np.mean(seed_accs)
        mean_auc = np.nanmean(seed_aucs)
        results.append({"name": name, "acc": mean_acc, "auc": mean_auc,
                        "acc_std": np.std(seed_accs), "n_seeds": len(SEEDS)})
        print(f"  {name:20s} acc={mean_acc:.3f}±{np.std(seed_accs):.3f} "
              f"auc={mean_auc:.3f} ({len(SEEDS)} seeds)")

    accs = [r["acc"] for r in results]
    aucs = [r["auc"] for r in results if not np.isnan(r["auc"])]
    print(f"  {'MEAN':20s} acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f} "
          f"(over {len(results)} datasets)")
    return results


def main():
    configs = [
        ("c1 base (k1@12L)", REPO / "ckpt_c1_base/step-80000.ckpt", 12, 1),
        ("c2 loop (k2@12L)", REPO / "ckpt_c2_loop/step-80000.ckpt", 12, 2),
        ("c3 deep (k1@24L)", REPO / "ckpt_c3_deep/step-80000.ckpt", 24, 1),
    ]

    all_results = {}
    for tag, ckpt, nlayers, loop_k in configs:
        model = build_model(ckpt, nlayers, loop_k)
        all_results[tag] = evaluate_all(model, tag)
        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}\nSUMMARY (mean across datasets, {len(SEEDS)} seeds each)\n{'='*60}")
    print(f"{'config':>25s}  {'acc':>6s}  {'auc':>6s}")
    for tag in all_results:
        accs = [r["acc"] for r in all_results[tag]]
        aucs = [r["auc"] for r in all_results[tag] if not np.isnan(r["auc"])]
        print(f"{tag:>25s}  {np.mean(accs):.4f}  {np.mean(aucs):.4f}")

    # Win-rate: per-dataset, c2 vs c1 and c2 vs c3 (win / tie / loss)
    tags = list(all_results.keys())
    if len(tags) == 3:
        r1, r2, r3 = all_results[tags[0]], all_results[tags[1]], all_results[tags[2]]
        names = [r["name"] for r in r1]
        n = len(names)
        TOL = 1e-9

        def wtl(a, b):  # a vs b: (win, tie, loss)
            w = sum(1 for i in range(n) if a[i]["acc"] > b[i]["acc"] + TOL)
            t = sum(1 for i in range(n) if abs(a[i]["acc"] - b[i]["acc"]) <= TOL)
            return w, t, n - w - t

        print(f"\nWIN / TIE / LOSS (accuracy, exact ties counted):")
        for label, a, b in [("c2 vs c1", r2, r1), ("c2 vs c3", r2, r3), ("c1 vs c3", r1, r3)]:
            w, t, l = wtl(a, b)
            print(f"  {label}: W={w} T={t} L={l} | win={100*w/n:.0f}% | "
                  f"win+tie={100*(w+t)/n:.0f}% ({w+t}/{n})")


if __name__ == "__main__":
    main()
