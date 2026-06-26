#!/usr/bin/env python
"""Intermediate-checkpoint TREND eval on TabArena (read trend mid-training).

Two questions:
  (A) TREND — does ROC-AUC improve with training steps? Each run at its NATIVE loop_k:
        loopk4 (k=4): steps 10k/20k/30k/40k
        loopk6 (k=6): steps 10k/20k
        curric  (native): 10k->k1, 20k->k2, 30k->k3, 40k->k4
                 (curriculum stage boundaries are 13333/26666/39999, so these
                  checkpoints land one-per-stage and the native trend spans k=1..4)
  (B) CROSS-K (binary-classifier prerequisite) — take step-40000 of BOTH the
        curriculum run and the fixed-loopk4 run, eval each at loop_k=1,2,3,4.
        Hypothesis: the curriculum model holds across ALL k (every k was trained),
        while the fixed-k model degenerates off its trained k=4. If the curriculum
        model holds, adaptive per-row depth selection becomes viable.

Reuses build_model / predict_proba / load_task / score / gbdt_proba from the
existing TabArena scripts so behavior is byte-identical to the c1/c2/c3 eval.
Builds one model instance PER (ckpt, loop_k) — loop_k is bound per-instance
(contamination-proof), so loading many configs in one process is safe.
"""

from __future__ import annotations

import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from run_benchmark_eval import build_model, predict_proba          # noqa: E402
from run_tabarena_eval import load_task, score, gbdt_proba         # noqa: E402

CKPT = os.path.join(REPRO, "ckpt")
TASKS_JSON = os.path.join(HERE, "tabarena_tasks.json")
OUT = os.path.join(REPRO, "trend_results.json")

# __APPEND_MARKER__


def ck(run, step):
    return os.path.join(CKPT, run, f"step-{step}.ckpt")


def build_configs():
    """(tag, ckpt_path, nlayers, loop_k, group). Only include checkpoints that exist."""
    cfgs = []
    # (A) TREND at native k
    for s in (10000, 20000, 30000, 40000):
        cfgs.append((f"loopk4_s{s//1000}k_k4", ck("loopk4_12L_80000", s), 12, 4, "trend_k4"))
    for s in (10000, 20000, 30000, 40000):
        cfgs.append((f"loopk6_s{s//1000}k_k6", ck("loopk6_12L_80000", s), 12, 6, "trend_k6"))
    curric_native = {10000: 1, 20000: 2, 30000: 3, 40000: 4}
    for s, k in curric_native.items():
        cfgs.append((f"curric_s{s//1000}k_k{k}", ck("curric_k1to6_12L_80000", s), 12, k, "trend_curric"))
    # (B) CROSS-K at step-40000 (k=4 instances already built above; add k=1,2,3)
    for k in (1, 2, 3):
        cfgs.append((f"curric_s40k_k{k}", ck("curric_k1to6_12L_80000", 40000), 12, k, "crossk_curric"))
        cfgs.append((f"loopk4_s40k_k{k}", ck("loopk4_12L_80000", 40000), 12, k, "crossk_loopk4"))
    # keep only existing checkpoints
    out = [c for c in cfgs if os.path.exists(c[1])]
    missing = [c[0] for c in cfgs if not os.path.exists(c[1])]
    if missing:
        print(f"[skip missing ckpt] {missing}", flush=True)
    return out


def main():
    import torch  # noqa: F401
    tasks = json.load(open(TASKS_JSON))
    clf = sorted([r for r in tasks if 2 <= r["cls"] <= 10], key=lambda r: r["rows"])
    print(f"TabArena clf datasets (<=10 cls): {len(clf)}", flush=True)

    configs = build_configs()
    print(f"building {len(configs)} model instances...", flush=True)
    t0 = time.time()
    models = {}
    for tag, path, nl, k, grp in configs:
        models[tag] = build_model(path, nl, k)
    # also tag-print the curriculum checkpoint metadata to confirm stage mapping
    import torch as _t
    for run, step in [("curric_k1to6_12L_80000", 40000)]:
        c = _t.load(ck(run, step), map_location="cpu", weights_only=False)
        print(f"[ckpt meta] {run}/step-{step}: stage={c.get('curriculum_stage')} "
              f"loop_k={c.get('curriculum_loop_k')} curr_step={c.get('curr_step')}", flush=True)
    print(f"built {len(models)} models in {time.time()-t0:.0f}s", flush=True)

    tags = [c[0] for c in configs]
    results = {tag: {} for tag in tags}
    results["catboost"] = {}; results["xgboost"] = {}

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
        for gb in ("catboost", "xgboost"):
            try:
                p = gbdt_proba(gb, Xtr, ytr, Xte, ncls)
                a, u = score(yte, p, ncls)
            except Exception:
                a, u = float("nan"), float("nan")
            results[gb][name] = (a, u)
        print(f"  done {name[:34]:34s} (n={len(Xtr)},f={Xtr.shape[1]},c={ncls})", flush=True)

    json.dump({"configs": [(t, p, nl, k, g) for t, p, nl, k, g in configs],
               "results": results}, open(OUT, "w"), indent=1)
    summarize(results, clf, configs)


def _mean_auc(results, tag, names):
    v = [results[tag][n][1] for n in names if n in results[tag] and not np.isnan(results[tag][n][1])]
    return (np.mean(v), len(v)) if v else (float("nan"), 0)


def summarize(results, clf, configs):
    names = [r["name"] for r in clf if r["name"] in results.get("catboost", {})]
    cat, _ = _mean_auc(results, "catboost", names)
    xgb, _ = _mean_auc(results, "xgboost", names)
    print(f"\n{'='*64}\nTREND — mean ROC-AUC over {len(names)} datasets\n{'='*64}")
    print(f"  [baselines]  catboost={cat:.4f}  xgboost={xgb:.4f}")

    def block(title, prefix_tags):
        print(f"\n-- {title} --")
        for tag in prefix_tags:
            m, n = _mean_auc(results, tag, names)
            print(f"  {tag:18s}  AUC={m:.4f}  (n={n})")

    block("(A) loopk4 trend @ k=4",  [c[0] for c in configs if c[4] == "trend_k4"])
    block("(A) loopk6 trend @ k=6",  [c[0] for c in configs if c[4] == "trend_k6"])
    block("(A) curric trend @ native k", [c[0] for c in configs if c[4] == "trend_curric"])

    # (B) cross-k tables at step-40000
    print(f"\n{'='*64}\n(B) CROSS-K @ step-40000 — does the model hold across loop_k?\n{'='*64}")
    print("  loop_k:            k1      k2      k3      k4")
    def crossk_row(label, run_step_prefix):
        cells = []
        for k in (1, 2, 3, 4):
            tag = f"{run_step_prefix}_k{k}"
            m, _ = _mean_auc(results, tag, names)
            cells.append(f"{m:.4f}" if not np.isnan(m) else "  -   ")
        print(f"  {label:16s} " + "  ".join(cells))
    crossk_row("curriculum", "curric_s40k")
    crossk_row("fixed-loopk4", "loopk4_s40k")
    print("\n  (curriculum trained on k=1..4 by step-40k; fixed-loopk4 trained only at k=4.")
    print("   If curriculum stays flat across k while fixed-loopk4 degenerates off k=4,")
    print("   adaptive per-row depth is viable.)")


if __name__ == "__main__":
    main()
