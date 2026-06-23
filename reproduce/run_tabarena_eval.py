#!/usr/bin/env python
"""TabArena-aligned eval of the 3 looped checkpoints (c1/c2/c3).

Aligns with TabPFN-3's protocol AS FAITHFULLY AS our small model allows:
  * REAL benchmark: the official TabArena-v0.1 classification suite (38 datasets,
    all <=10 classes so our max_classes=10 model can run them), pulled from OpenML.
  * OFFICIAL splits: OpenML task train/test split indices (fold 0), not ad-hoc.
  * PRIMARY metric: ROC-AUC (OvR macro), TabPFN-3's headline metric. + accuracy.
  * BASELINES: CatBoost + XGBoost (defaults) for external context.
  * Per-dataset normalized score (vs the per-dataset best) for an Elo-flavored aggregate.

DOCUMENTED DEVIATIONS from full TabArena (our model is tiny, trained seq_len<=600):
  * Train CONTEXT capped at CTX_CAP rows (subsample) — our model can't ingest 150k rows.
    This is the in-context budget, matching how the model was trained, NOT TabArena's
    full-data regime. So absolute numbers are NOT comparable to the TabArena leaderboard;
    the comparison that IS valid is c1-vs-c2-vs-c3 (identical budget) and vs GBDT on the
    SAME capped context.
  * Single fold (fold 0) instead of TabArena's 3-10 repeats x folds (compute).
  * Features capped at FEAT_CAP (our model's regime); high-dim sets (Bioresponse 1777,
    hiva 1618, kddcup 213, MIC 112) are flagged — feature subsample applied.
"""

from __future__ import annotations

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, "/data/zxiebk/workspace/train/PFN/TabPFN/reproduce")

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import openml

from run_benchmark_eval import build_model, predict_proba

DEVICE = "cuda"
REPO = "/data/zxiebk/workspace/train/PFN/TabPFN/reproduce"
CTX_CAP = 1000     # train-context budget (model trained on <=600; 1000 is a mild stretch)
TEST_CAP = 500     # test rows scored per dataset (cap for speed)
FEAT_CAP = 40      # feature budget (model trained on <=40 features)
SEED = 42

# __APPEND_MARKER__


def load_task(task_id):
    """Load an OpenML task: returns (Xtr, ytr, Xte, yte) using the official fold-0 split,
    numeric-encoded, NaN-imputed, feature/context capped."""
    t = openml.tasks.get_task(task_id, download_splits=True)
    ds = t.get_dataset()
    X, y, _, _ = ds.get_data(target=t.target_name, dataset_format="dataframe")
    # numeric encode features (one-hot-free: factorize categoricals)
    Xn = np.zeros((len(X), X.shape[1]), dtype=np.float32)
    for j, col in enumerate(X.columns):
        s = X[col]
        if s.dtype.kind in "biufc":
            Xn[:, j] = s.to_numpy(dtype=np.float32)
        else:
            Xn[:, j] = s.astype("category").cat.codes.to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(y.astype(str)).astype(np.int64)
    # official split FIRST (fold 0, repeat 0), then impute using TRAIN stats only
    # (computing column means over train+test would leak test info into the imputation).
    tr_idx, te_idx = t.get_train_test_split_indices(fold=0, repeat=0)
    Xtr, ytr, Xte, yte = Xn[tr_idx], y[tr_idx], Xn[te_idx], y[te_idx]
    col_mean = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    for arr in (Xtr, Xte):
        bad = np.where(~np.isfinite(arr))
        arr[bad] = np.take(col_mean, bad[1])
    # feature cap: keep top-FEAT_CAP by TRAIN variance (test must not influence selection)
    if Xtr.shape[1] > FEAT_CAP:
        var = Xtr.var(0)
        keep = np.argsort(-var)[:FEAT_CAP]
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    # context cap: stratified-ish subsample of train
    rng = np.random.RandomState(SEED)
    if len(Xtr) > CTX_CAP:
        idx = rng.choice(len(Xtr), CTX_CAP, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > TEST_CAP:
        idx = rng.choice(len(Xte), TEST_CAP, replace=False)
        Xte, yte = Xte[idx], yte[idx]
    return Xtr, ytr, Xte, yte


def score(yte, proba, n_classes):
    pred = proba.argmax(1)
    acc = accuracy_score(yte, pred)
    # renormalize: multiclass roc_auc requires rows sum to exactly 1.0 (softmax+pad
    # leaves 0.9999 from float precision -> ValueError without this)
    proba = proba / proba.sum(1, keepdims=True).clip(min=1e-12)
    try:
        if n_classes == 2:
            auc = roc_auc_score(yte, proba[:, 1])
        else:
            auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                                labels=list(range(proba.shape[1])))
    except ValueError:
        auc = float("nan")
    return acc, auc


def gbdt_proba(kind, Xtr, ytr, Xte, n_classes):
    if kind == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                               verbose=0, thread_count=4, allow_writing_files=False)
    else:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          tree_method="hist", n_jobs=4, verbosity=0)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    if p.shape[1] < n_classes:
        pad = np.zeros((len(p), n_classes - p.shape[1])); p = np.concatenate([p, pad], 1)
    return p


def main():
    tasks = json.load(open(f"{REPO}/tabarena_tasks.json"))
    clf = [r for r in tasks if r["cls"] >= 2 and r["cls"] <= 10]
    clf = sorted(clf, key=lambda r: r["rows"])
    print(f"TabArena classification datasets (<=10 cls): {len(clf)}", flush=True)

    configs = [
        ("c1_base", f"{REPO}/ckpt_c1_base/step-80000.ckpt", 12, 1),
        ("c2_loop", f"{REPO}/ckpt_c2_loop/step-80000.ckpt", 12, 2),
        ("c3_deep", f"{REPO}/ckpt_c3_deep/step-80000.ckpt", 24, 1),
    ]
    import looped_step2 as L
    models = {tag: build_model(c, nl, lk) for tag, c, nl, lk in configs}
    # loop_k is now bound per-model-instance inside build_model (set_loop_on_model),
    # so no global state to juggle — loading all configs in one process is safe.

    results = {tag: {} for tag, *_ in configs}
    results["catboost"] = {}; results["xgboost"] = {}

    for r in clf:
        name, tid, ncls = r["name"], r["task"], r["cls"]
        try:
            Xtr, ytr, Xte, yte = load_task(tid)
        except Exception as e:
            print(f"  SKIP {name}: {str(e)[:60]}", flush=True); continue
        ncls_eff = int(max(ytr.max(), yte.max())) + 1
        line = f"  {name[:34]:34s} (n={len(Xtr)},f={Xtr.shape[1]},c={ncls_eff})"
        for tag, *_ in configs:
            try:
                p = predict_proba(models[tag], Xtr, ytr, Xte)
                a, u = score(yte, p, ncls_eff)
            except Exception as e:
                a, u = float("nan"), float("nan")
            results[tag][name] = (a, u)
            line += f" | {tag[:2]} auc={u:.3f}"
        for gb in ["catboost", "xgboost"]:
            try:
                p = gbdt_proba(gb, Xtr, ytr, Xte, ncls_eff)
                a, u = score(yte, p, ncls_eff)
            except Exception:
                a, u = float("nan"), float("nan")
            results[gb][name] = (a, u)
        line += f" | cat={results['catboost'][name][1]:.3f} xgb={results['xgboost'][name][1]:.3f}"
        print(line, flush=True)

    json.dump(results, open(f"{REPO}/tabarena_results.json", "w"), indent=1)
    summarize(results, clf)


def summarize(results, clf):
    names = [r["name"] for r in clf if r["name"] in results["c1_base"]]
    methods = ["c1_base", "c2_loop", "c3_deep", "catboost", "xgboost"]
    print(f"\n{'='*70}\nSUMMARY — mean ROC-AUC over {len(names)} TabArena clf datasets\n{'='*70}")
    means = {}
    for m in methods:
        aucs = [results[m][n][1] for n in names if not np.isnan(results[m][n][1])]
        accs = [results[m][n][0] for n in names if not np.isnan(results[m][n][0])]
        means[m] = np.mean(aucs)
        print(f"  {m:10s}  AUC={np.mean(aucs):.4f}  ACC={np.mean(accs):.4f}  (n={len(aucs)})")
    # normalized score (per-dataset min-max over methods) — Elo-flavored
    print(f"\nNormalized AUC (per-dataset min-max across methods, higher=better):")
    norm = {m: [] for m in methods}
    for n in names:
        vals = {m: results[m][n][1] for m in methods if not np.isnan(results[m][n][1])}
        if len(vals) < 2: continue
        lo, hi = min(vals.values()), max(vals.values())
        for m in vals:
            norm[m].append((vals[m]-lo)/(hi-lo) if hi > lo else 1.0)
    for m in methods:
        print(f"  {m:10s}  norm={np.mean(norm[m]):.4f}")
    # win/tie/loss c2 vs others
    print(f"\nWIN/TIE/LOSS (ROC-AUC, c2_loop vs each):")
    for opp in ["c1_base", "c3_deep", "catboost", "xgboost"]:
        w=t=l=0
        for n in names:
            a, b = results["c2_loop"][n][1], results[opp][n][1]
            if np.isnan(a) or np.isnan(b): continue
            if a > b+1e-9: w+=1
            elif abs(a-b)<=1e-9: t+=1
            else: l+=1
        tot=w+t+l
        print(f"  c2 vs {opp:10s}: W={w} T={t} L={l} | win+tie={100*(w+t)/tot:.0f}%")


if __name__ == "__main__":
    main()
