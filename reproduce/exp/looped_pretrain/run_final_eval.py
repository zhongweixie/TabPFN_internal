#!/usr/bin/env python
"""Final-checkpoint eval: loopk4/loopk6/curric step-80000 vs c1/c2/c3 baselines.

Reuses GBDT results from trend_results.json (already computed, not re-run).
Designed to be re-run as more step-80000 checkpoints become available.

Two questions:
  (A) Final AUC: how does loopk4-80k (and later loopk6/curric) compare to c1/c2/c3?
  (B) Cross-k @ 80k: loopk4 and curric at loop_k=1,2,3,4 — does degeneration hold
      at full training? Is curric still flat?
"""
from __future__ import annotations
import sys, os, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from run_benchmark_eval import build_model, predict_proba
from run_tabarena_eval import load_task, score

TASKS_JSON = os.path.join(HERE, "tabarena_tasks.json")
CKPT = os.path.join(REPRO, "ckpt")
OUT = os.path.join(REPRO, "final_eval_results.json")
GBDT_CACHE = os.path.join(REPRO, "trend_results.json")   # pre-computed catboost/xgboost

# __APPEND_MARKER__


def ck(run, step=80000):
    return os.path.join(CKPT, run, f"step-{step}.ckpt")


def build_configs():
    """Return (tag, ckpt_path, nlayers, loop_k, group). Skip missing ckpts."""
    cfgs = [
        # (A) final AUC at native k
        ("c1_base_80k",  f"{REPRO}/ckpt_c1_base/step-80000.ckpt",  12, 1, "final"),
        ("c2_loop_80k",  f"{REPRO}/ckpt_c2_loop/step-80000.ckpt",  12, 2, "final"),
        ("c3_deep_80k",  f"{REPRO}/ckpt_c3_deep/step-80000.ckpt",  24, 1, "final"),
        ("loopk4_80k_k4", ck("loopk4_12L_80000"), 12, 4, "final"),
        ("loopk6_80k_k6", ck("loopk6_12L_80000"), 12, 6, "final"),
        ("curric_80k_k6", ck("curric_k1to6_12L_80000"), 12, 6, "final"),
        # (B) cross-k at 80k
        *[("loopk4_80k_k%d" % k, ck("loopk4_12L_80000"), 12, k, "crossk_loopk4")
          for k in (1, 2, 3)],
        *[("curric_80k_k%d" % k, ck("curric_k1to6_12L_80000"), 12, k, "crossk_curric")
          for k in (1, 2, 3, 4, 5)],
    ]
    out = [c for c in cfgs if os.path.exists(c[1])]
    miss = [c[0] for c in cfgs if not os.path.exists(c[1])]
    if miss:
        print(f"[skip missing] {miss}", flush=True)
    return out


def main():
    import torch  # noqa: F401 (triggers CUDA init early)
    tasks = json.load(open(TASKS_JSON))
    clf = sorted([r for r in tasks if 2 <= r["cls"] <= 10], key=lambda r: r["rows"])
    print(f"datasets: {len(clf)}", flush=True)

    configs = build_configs()
    print(f"building {len(configs)} models...", flush=True)
    models = {tag: build_model(p, nl, k) for tag, p, nl, k, _ in configs}

    # load pre-computed GBDT results
    gbdt = {}
    if os.path.exists(GBDT_CACHE):
        raw = json.load(open(GBDT_CACHE))["results"]
        gbdt = {g: raw[g] for g in ("catboost", "xgboost") if g in raw}
        print(f"loaded GBDT cache for {len(next(iter(gbdt.values())))} datasets", flush=True)

    tags = [c[0] for c in configs]
    results = {tag: {} for tag in tags}

    for r in clf:
        name, tid = r["name"], r["task"]
        try:
            Xtr, ytr, Xte, yte = load_task(tid)
        except Exception as e:
            print(f"  SKIP {name}: {str(e)[:60]}", flush=True); continue
        ncls = int(max(ytr.max(), yte.max())) + 1
        for tag in tags:
            try:
                p = predict_proba(models[tag], Xtr, ytr, Xte)
                a, u = score(yte, p, ncls)
            except Exception:
                a, u = float("nan"), float("nan")
            results[tag][name] = (a, u)
        print(f"  {name[:36]:36s} n={len(Xtr)} f={Xtr.shape[1]} c={ncls}", flush=True)

    json.dump({"configs": [(t, p, nl, k, g) for t, p, nl, k, g in configs],
               "results": {**results, **gbdt}}, open(OUT, "w"), indent=1)
    summarize(results, gbdt, clf, configs)


def _mean(results, tag, names):
    v = [results[tag][n][1] for n in names if n in results.get(tag, {}) and not np.isnan(results[tag][n][1])]
    return np.mean(v) if v else float("nan")


def summarize(results, gbdt, clf, configs):
    names = [r["name"] for r in clf if r["name"] in next(iter(results.values()), {})]
    cat = _mean(gbdt, "catboost", names)
    xgb = _mean(gbdt, "xgboost", names)

    print(f"\n{'='*60}\n(A) FINAL AUC @ 80k (mean ROC-AUC, {len(names)} datasets)\n{'='*60}")
    print(f"  [baselines] catboost={cat:.4f}  xgboost={xgb:.4f}")
    for tag, *_, grp in configs:
        if grp != "final": continue
        m = _mean(results, tag, names)
        print(f"  {tag:20s}  {m:.4f}")

    print(f"\n{'='*60}\n(B) CROSS-K @ 80k\n{'='*60}")
    print("  loop_k:            k1      k2      k3      k4      k5      k6")
    for label, prefix in [("curric_80k",   "curric_80k"),
                           ("loopk4_80k",  "loopk4_80k")]:
        cells = []
        for k in (1, 2, 3, 4, 5, 6):
            tag = f"{prefix}_k{k}"
            m = _mean(results, tag, names)
            cells.append(f"{m:.4f}" if not np.isnan(m) else "  -   ")
        print(f"  {label:16s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
