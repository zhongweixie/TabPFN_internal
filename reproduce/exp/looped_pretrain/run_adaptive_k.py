#!/usr/bin/env python
"""Per-row adaptive loop_k selector: train a lightweight classifier that picks
which loop depth to use for each test row, then evaluate vs fixed-k baselines.

Two bases:
  curric step-60K: k=3..6 all flat ~0.808 (best multi-k checkpoint)
  curric step-80K: k=6 specialist (0.810), k=1..5 degraded (0.742-0.766)

Design per dataset:
  1. Split train → ctx (80%) + sel (20%, ≥50 rows)
  2. Run model at each k on sel rows (ctx as context) → per-row probas
  3. best_k_i = argmax_k P(y_true_i | model at k)  (supervised on known labels)
  4. Train LogisticRegression(sel features → best_k label)
  5. For test rows: predict k via selector → run model at that k → AUC
  6. Also compute: oracle AUC (argmax over k using test true labels = upper bound)
                   fixed-k AUC at each k
                   loopk4-80K reference (fixed specialist 0.8174)

Key bar to beat: loopk4 fixed specialist = 0.8174
"""
from __future__ import annotations
import sys, os, json, numpy as np, time

HERE  = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from run_benchmark_eval import build_model, predict_proba
from run_tabarena_eval  import load_task, score

TASKS_JSON = os.path.join(HERE, "tabarena_tasks.json")
CKPT       = os.path.join(REPRO, "ckpt")
OUT_JSON   = os.path.join(REPRO, "indexer_gate_run.out")   # log
RESULTS    = os.path.join(REPRO, "adaptive_k_results.json")

LOOPK4_REF_AUC = 0.8174   # reference from final_eval (same GBDT-reuse protocol)

SEED = 42

# __APPEND_MARKER__


def ck(run, step):
    return os.path.join(CKPT, run, f"step-{step}.ckpt")


def build_all_models():
    """Build all model instances needed for both bases."""
    models = {}
    # curric step-60K: k=3,4,5,6
    for k in (3, 4, 5, 6):
        tag = f"c60k_k{k}"
        models[tag] = build_model(ck("curric_k1to6_12L_80000", 60000), 12, k)
        print(f"  built {tag}", flush=True)
    # curric step-80K: k=1..6
    for k in (1, 2, 3, 4, 5, 6):
        tag = f"c80k_k{k}"
        models[tag] = build_model(ck("curric_k1to6_12L_80000", 80000), 12, k)
        print(f"  built {tag}", flush=True)
    # loopk4-80K reference (fixed specialist)
    models["loopk4_k4"] = build_model(ck("loopk4_12L_80000", 80000), 12, 4)
    print(f"  built loopk4_k4 (reference)", flush=True)
    return models


def ctx_sel_split(Xtr, ytr, min_sel=50):
    """Split train into ctx (80%) and sel (20%, ≥ min_sel) for selector training."""
    n = len(Xtr)
    n_sel = max(min_sel, n // 5)
    if n_sel >= n:          # too small: use 50/50
        n_sel = n // 2
    n_ctx = n - n_sel
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    ctx_idx, sel_idx = idx[:n_ctx], idx[n_ctx:]
    return Xtr[ctx_idx], ytr[ctx_idx], Xtr[sel_idx], ytr[sel_idx]


def train_selector(Xsel, ysel, sel_probas_dict, ks):
    """Train a LogisticRegression to predict best k from row features.
    sel_probas_dict: {k: array(n_sel, n_classes)}
    Returns: fitted selector (or None if trivially one k).
    """
    from sklearn.linear_model import LogisticRegression
    n_sel = len(Xsel)
    n_classes = sel_probas_dict[ks[0]].shape[1]

    best_ks = []
    for i in range(n_sel):
        yi = ysel[i]
        if yi >= n_classes:
            # label out of range for this model (rare edge case): pick middle k
            best_ks.append(ks[len(ks)//2])
        else:
            # pick the k that gave highest probability for the true class
            best_k = max(ks, key=lambda k: sel_probas_dict[k][i, yi]
                         if yi < sel_probas_dict[k].shape[1] else 0.0)
            best_ks.append(best_k)

    best_ks = np.array(best_ks)
    # if all sel rows have the same best_k, LogisticRegression will fail
    unique = np.unique(best_ks)
    if len(unique) == 1:
        return None, best_ks[0]   # constant selector

    sel = LogisticRegression(max_iter=500, C=1.0, random_state=SEED,
                             multi_class="multinomial", solver="lbfgs")
    sel.fit(Xsel, best_ks)
    return sel, None


def adaptive_eval_dataset(models, ks, Xtr, ytr, Xte, yte):
    """Run adaptive-k eval on one dataset. Returns dict of metrics."""
    Xctx, yctx, Xsel, ysel = ctx_sel_split(Xtr, ytr)
    n_classes = int(max(ytr.max(), yte.max())) + 1

    # Get sel and test probas at each k
    sel_probas  = {k: predict_proba(models[k], Xctx, yctx, Xsel) for k in ks}
    test_probas = {k: predict_proba(models[k], Xctx, yctx, Xte)  for k in ks}

    # Train selector on sel rows
    selector, const_k = train_selector(Xsel, ysel, sel_probas, ks)

    # Adaptive: predict k for each test row
    if selector is None:
        pred_ks = np.full(len(Xte), const_k)
    else:
        pred_ks = selector.predict(Xte)

    adap_proba = np.array([test_probas[k][i] for i, k in enumerate(pred_ks)])
    adap_proba = adap_proba / adap_proba.sum(1, keepdims=True).clip(min=1e-12)
    _, adap_auc = score(yte, adap_proba, n_classes)

    # Oracle: pick best k per test row using true labels
    oracle_proba = []
    for i in range(len(Xte)):
        yi = yte[i]
        best_k = max(ks, key=lambda k: test_probas[k][i, yi]
                     if yi < test_probas[k].shape[1] else 0.0)
        oracle_proba.append(test_probas[best_k][i])
    oracle_proba = np.array(oracle_proba)
    oracle_proba = oracle_proba / oracle_proba.sum(1, keepdims=True).clip(min=1e-12)
    _, oracle_auc = score(yte, oracle_proba, n_classes)

    # Fixed-k at each k
    fixed_aucs = {}
    for k in ks:
        p = test_probas[k] / test_probas[k].sum(1, keepdims=True).clip(min=1e-12)
        _, u = score(yte, p, n_classes)
        fixed_aucs[k] = u

    # k distribution selected by the selector
    k_counts = {k: int((pred_ks == k).sum()) for k in ks}

    return dict(adaptive=adap_auc, oracle=oracle_auc, fixed=fixed_aucs,
                k_dist=k_counts, n_sel=len(Xsel), n_test=len(Xte))


def run_base(name, prefix, ks, models, clf, model_key_fn):
    """Evaluate one base (60K or 80K) across all datasets."""
    print(f"\n{'='*60}", flush=True)
    print(f"BASE: {name}  |  k-range: {ks}", flush=True)
    print(f"{'='*60}", flush=True)

    base_models = {k: models[model_key_fn(k)] for k in ks}
    loopk4 = models["loopk4_k4"]

    per_ds = {}
    for r in clf:
        name_ds, tid = r["name"], r["task"]
        try:
            Xtr, ytr, Xte, yte = load_task(tid)
        except Exception as e:
            print(f"  SKIP {name_ds}: {str(e)[:60]}", flush=True)
            continue

        try:
            res = adaptive_eval_dataset(base_models, ks, Xtr, ytr, Xte, yte)
        except Exception as e:
            print(f"  ERR  {name_ds}: {str(e)[:60]}", flush=True)
            continue

        # loopk4 reference on same ctx split (full train as context for fair compare)
        try:
            n_classes = int(max(ytr.max(), yte.max())) + 1
            p4 = predict_proba(loopk4, Xtr, ytr, Xte)
            p4 = p4 / p4.sum(1, keepdims=True).clip(min=1e-12)
            _, ref4 = score(yte, p4, n_classes)
        except Exception:
            ref4 = float("nan")

        res["loopk4_ref"] = ref4
        per_ds[name_ds] = res

        best_fixed = max(res["fixed"].values())
        print(f"  {name_ds[:32]:32s} "
              f"oracle={res['oracle']:.4f} adap={res['adaptive']:.4f} "
              f"best_fixed={best_fixed:.4f} lk4={ref4:.4f} "
              f"k_dist={res['k_dist']}", flush=True)

    return per_ds


def summarize(name, per_ds, ks):
    names = [n for n in per_ds if not np.isnan(per_ds[n]["adaptive"])]
    def mn(key): return np.nanmean([per_ds[n][key] for n in names])
    def mn_fixed(k): return np.nanmean([per_ds[n]["fixed"][k] for n in names
                                        if k in per_ds[n]["fixed"]])

    print(f"\n--- SUMMARY: {name} ---")
    print(f"  oracle (upper bound):      {mn('oracle'):.4f}")
    print(f"  adaptive (selector):       {mn('adaptive'):.4f}")
    for k in ks:
        print(f"  fixed k={k}:               {mn_fixed(k):.4f}")
    print(f"  loopk4 ref (same ctx):     {mn('loopk4_ref'):.4f}")
    print(f"  loopk4 global 80K:         {LOOPK4_REF_AUC:.4f}  (full-train ctx)")
    best_fixed = max(mn_fixed(k) for k in ks)
    adap = mn("adaptive")
    print(f"\n  adaptive vs best_fixed_k:  {adap-best_fixed:+.4f}")
    print(f"  oracle  vs best_fixed_k:   {mn('oracle')-best_fixed:+.4f}")
    print(f"  adaptive vs loopk4-80K:    {adap-LOOPK4_REF_AUC:+.4f}")


def main():
    import torch  # noqa
    tasks = json.load(open(TASKS_JSON))
    clf   = sorted([r for r in tasks if 2 <= r["cls"] <= 10], key=lambda r: r["rows"])
    print(f"datasets: {len(clf)}", flush=True)

    print("building models...", flush=True)
    t0 = time.time()
    models = build_all_models()
    print(f"built {len(models)} models in {time.time()-t0:.0f}s", flush=True)

    all_results = {}

    # Base A: curric step-60K, k=3..6
    ks_60 = [3, 4, 5, 6]
    res_60 = run_base("curric-60K", "c60k", ks_60, models,
                      model_key_fn=lambda k: f"c60k_k{k}", clf=clf)
    summarize("curric-60K (k=3..6)", res_60, ks_60)
    all_results["curric_60k"] = {n: {**v, "fixed": {str(k):float(u) for k,u in v["fixed"].items()},
                                     "k_dist": {str(k):int(c) for k,c in v["k_dist"].items()}}
                                 for n,v in res_60.items()}

    # Base B: curric step-80K, k=1..6
    ks_80 = [1, 2, 3, 4, 5, 6]
    res_80 = run_base("curric-80K", "c80k", ks_80, models,
                      model_key_fn=lambda k: f"c80k_k{k}", clf=clf)
    summarize("curric-80K (k=1..6)", res_80, ks_80)
    all_results["curric_80k"] = {n: {**v, "fixed": {str(k):float(u) for k,u in v["fixed"].items()},
                                     "k_dist": {str(k):int(c) for k,c in v["k_dist"].items()}}
                                 for n,v in res_80.items()}

    json.dump(all_results, open(RESULTS, "w"), indent=1)
    print(f"\nresults saved → {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
