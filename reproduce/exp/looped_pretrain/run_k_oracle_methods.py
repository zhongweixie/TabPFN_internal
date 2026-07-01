#!/usr/bin/env python
"""Explore multiple strategies for finding optimal loop_k.

Base: curric step-60K (k=3..6 flat at ~0.808).
All methods evaluated on TabArena 38-clf ROC-AUC.

Methods:
  A1. val-set k sweep       — split train 80/20, pick k with best val AUC
  A2. 3-fold CV k sweep     — CV on train, pick k with best mean fold AUC
  A3. dataset meta-features — n_train,n_feat,n_cls → predict best k (LightGBM)
  B1. output entropy @k=3   — if model already confident, use k=3; else k=6
  B2. output stability      — per-row: use k where output stopped changing
  C1. equal blend           — average proba across k=3..6
  C2. val-weighted blend    — learn blend weights on val set

Gold baselines:
  oracle   — per-row best k using test labels (unreachable ceiling)
  loopk4   — fixed loopk4 specialist (0.8174, the bar to beat)
  best_fixed — best single k from curric-60K
"""
from __future__ import annotations
import sys, os, json, numpy as np, time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

HERE  = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from run_benchmark_eval import build_model, predict_proba
from run_tabarena_eval  import load_task, score

TASKS_JSON  = os.path.join(HERE, "tabarena_tasks.json")
CKPT        = os.path.join(REPRO, "ckpt")
RESULTS_OUT = os.path.join(REPRO, "indexer_honest_run.out")
KS          = [3, 4, 5, 6]
SEED        = 42

# __APPEND_MARKER__


def ck(run, step):
    return os.path.join(CKPT, run, f"step-{step}.ckpt")


def _auc(y, p, n_classes):
    p = p / p.sum(1, keepdims=True).clip(min=1e-12)
    try:
        if n_classes == 2:
            return roc_auc_score(y, p[:, 1])
        return roc_auc_score(y, p, multi_class="ovr", average="macro",
                             labels=list(range(p.shape[1])))
    except Exception:
        return float("nan")


def _split(X, y, frac=0.8, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n = max(1, int(len(X) * frac))
    return X[idx[:n]], y[idx[:n]], X[idx[n:]], y[idx[n:]]


def _cv_folds(X, y, n_folds=3, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, n_folds)
    for fi in range(n_folds):
        val_idx = folds[fi]
        tr_idx  = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        yield X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def dataset_meta(X, y):
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes)
    imb = counts.max() / counts.sum()
    feat_var = float(np.var(X))
    return [len(X), X.shape[1], n_classes, float(imb), feat_var]


def eval_dataset(models, Xtr, ytr, Xte, yte):
    """Run all methods for one dataset. Returns dict."""
    ncls = int(max(ytr.max(), yte.max())) + 1
    res = {}

    # -- pre-compute test probas at each k (used by all methods) --
    test_p = {k: predict_proba(models[k], Xtr, ytr, Xte) for k in KS}

    # gold: oracle
    oracle_p = []
    for i in range(len(Xte)):
        yi = yte[i]
        bk = max(KS, key=lambda k: test_p[k][i, yi] if yi < test_p[k].shape[1] else 0.)
        oracle_p.append(test_p[bk][i])
    res["oracle"] = _auc(yte, np.array(oracle_p), ncls)

    # gold: best single fixed k
    fixed = {k: _auc(yte, test_p[k], ncls) for k in KS}
    res["best_fixed_k"] = max(fixed.values())
    res["fixed"] = fixed

    # C1. equal blend
    blend = sum(test_p[k] for k in KS) / len(KS)
    res["blend_equal"] = _auc(yte, blend, ncls)

    # --- methods that require a val split ---
    Xctx, yctx, Xval, yval = _split(Xtr, ytr, frac=0.8)
    val_p = {k: predict_proba(models[k], Xctx, yctx, Xval) for k in KS}
    # rebuild test probas with same ctx split for fair comparison
    test_p2 = {k: predict_proba(models[k], Xctx, yctx, Xte) for k in KS}

    # A1. val-set k sweep (dataset-level)
    val_auc = {k: _auc(yval, val_p[k], ncls) for k in KS}
    best_k_val = max(KS, key=lambda k: val_auc[k])
    res["A1_val_sweep"] = _auc(yte, test_p2[best_k_val], ncls)
    res["A1_chosen_k"] = best_k_val

    # A2. 3-fold CV k sweep (dataset-level)
    cv_aucs = {k: [] for k in KS}
    for Xcv, ycv, Xfv, yfv in _cv_folds(Xtr, ytr, n_folds=3):
        for k in KS:
            pv = predict_proba(models[k], Xcv, ycv, Xfv)
            cv_aucs[k].append(_auc(yfv, pv, ncls))
    mean_cv = {k: np.nanmean(cv_aucs[k]) for k in KS}
    best_k_cv = max(KS, key=lambda k: mean_cv[k])
    res["A2_cv_sweep"] = _auc(yte, test_p2[best_k_cv], ncls)
    res["A2_chosen_k"] = best_k_cv
    res["A2_cv_aucs"]  = {k: float(v) for k, v in mean_cv.items()}

    # B1. output entropy signal (per-row): low entropy at k=3 → use k=3, else k=6
    ent3 = -(test_p2[3] * np.log(test_p2[3].clip(min=1e-9))).sum(1)
    ent_thresh = np.median(ent3)
    pred_ks_b1 = np.where(ent3 <= ent_thresh, 3, 6)
    b1_p = np.array([test_p2[k][i] for i, k in enumerate(pred_ks_b1)])
    res["B1_entropy"] = _auc(yte, b1_p, ncls)

    # B2. output stability (per-row): use k where |p_k - p_{k+1}| < threshold
    # heuristic: if output barely changes k=3→k=4, use k=3, else use k=6
    diff34 = np.abs(test_p2[3] - test_p2[4]).max(1)  # max change per row
    stab_thresh = np.median(diff34)
    pred_ks_b2 = np.where(diff34 <= stab_thresh, 3, 6)
    b2_p = np.array([test_p2[k][i] for i, k in enumerate(pred_ks_b2)])
    res["B2_stability"] = _auc(yte, b2_p, ncls)

    # C2. val-weighted blend (learn blend weights on val)
    # simple: weight each k by its val AUC (normalised)
    w = np.array([val_auc[k] for k in KS])
    w = (w - w.min() + 1e-6)  # shift positive
    w = w / w.sum()
    blend_w = sum(w[i] * test_p2[KS[i]] for i in range(len(KS)))
    res["C2_val_blend"] = _auc(yte, blend_w, ncls)

    return res, dataset_meta(Xtr, ytr), best_k_cv


def main():
    tasks = json.load(open(TASKS_JSON))
    clf   = sorted([r for r in tasks if 2 <= r["cls"] <= 10], key=lambda r: r["rows"])
    print(f"datasets: {len(clf)}", flush=True)

    print("building curric-60K models (k=3..6)...", flush=True)
    models = {k: build_model(ck("curric_k1to6_12L_80000", 60000), 12, k) for k in KS}
    loopk4_model = build_model(ck("loopk4_12L_80000", 80000), 12, 4)
    print("models ready", flush=True)

    METHODS = ["oracle","best_fixed_k","A1_val_sweep","A2_cv_sweep",
               "B1_entropy","B2_stability","blend_equal","C2_val_blend","loopk4_ref"]
    all_res = []
    meta_rows, best_ks = [], []

    for r in clf:
        name, tid = r["name"], r["task"]
        try:
            Xtr, ytr, Xte, yte = load_task(tid)
        except Exception as e:
            print(f"  SKIP {name}: {str(e)[:60]}", flush=True); continue

        try:
            res, meta, bk_cv = eval_dataset(models, Xtr, ytr, Xte, yte)
        except Exception as e:
            print(f"  ERR  {name}: {str(e)[:80]}", flush=True); continue

        # loopk4 reference (full train as context)
        try:
            ncls = int(max(ytr.max(), yte.max())) + 1
            p4 = predict_proba(loopk4_model, Xtr, ytr, Xte)
            res["loopk4_ref"] = _auc(yte, p4, ncls)
        except Exception:
            res["loopk4_ref"] = float("nan")

        all_res.append(res)
        meta_rows.append(meta)
        best_ks.append(bk_cv)

        line = f"  {name[:28]:28s}"
        for m in METHODS:
            v = res.get(m, float("nan"))
            line += f"  {m[:6]}={v:.3f}" if not np.isnan(v) else f"  {m[:6]}=  nan"
        print(line, flush=True)

    print(f"\n{'='*70}", flush=True)
    print("SUMMARY — mean ROC-AUC across datasets", flush=True)
    print(f"{'='*70}", flush=True)
    for m in METHODS:
        vals = [r[m] for r in all_res if not np.isnan(r.get(m, float("nan")))]
        mu = np.mean(vals) if vals else float("nan")
        vs_lk4 = mu - np.nanmean([r.get("loopk4_ref", float("nan")) for r in all_res])
        print(f"  {m:18s}  {mu:.4f}  (vs loopk4: {vs_lk4:+.4f})", flush=True)

    # A3: dataset meta-feature model — train on all-but-one, predict best k
    print(f"\n{'='*70}", flush=True)
    print("A3. Dataset meta-features → best_k (leave-one-dataset-out)", flush=True)
    try:
        import lightgbm as lgb
        X_meta = np.array(meta_rows)
        y_meta = np.array(best_ks)
        # leave-one-out cross validation
        meta_preds = []
        for i in range(len(X_meta)):
            X_tr = np.delete(X_meta, i, axis=0)
            y_tr = np.delete(y_meta, i)
            if len(np.unique(y_tr)) < 2:
                meta_preds.append(y_meta[i])  # trivial
                continue
            clf_lgb = lgb.LGBMClassifier(n_estimators=50, verbose=-1, random_state=SEED)
            clf_lgb.fit(X_tr, y_tr)
            meta_preds.append(clf_lgb.predict(X_meta[i:i+1])[0])
        match_rate = np.mean(np.array(meta_preds) == y_meta)
        print(f"  meta-feature match rate vs CV-best-k: {match_rate:.2f}", flush=True)

        # evaluate A3 on test sets using the predicted k
        a3_aucs = []
        for i, (meta_k, r) in enumerate(zip(meta_preds, all_res)):
            if meta_k in r["fixed"]:
                a3_aucs.append(r["fixed"][meta_k])
        print(f"  A3 mean AUC: {np.nanmean(a3_aucs):.4f}", flush=True)
    except ImportError:
        print("  lightgbm not available, skipping A3", flush=True)

    json.dump(all_res, open(RESULTS_OUT.replace(".out", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
