#!/usr/bin/env python
"""Comprehensive overfitting / catastrophic-forgetting diagnostic.

Evaluates loopk4, loopk6, and curriculum at checkpoints 40k-80k (every 10k),
each at multiple loop_k values, on TabArena 38-clf ROC-AUC.

Diagnostic questions:
  (1) OVERFITTING? If loopk4/loopk6 TabArena AUC is declining after ~60k, that's
      overfitting. If it's flat or improving, it's not.
  (2) CATASTROPHIC FORGETTING in curriculum? If curric's k=4 AUC drops specifically
      around step-53332 (stage-switch to k=5) or step-66665 (stage-switch to k=6)
      while k=6 AUC rises, that's forgetting. If k=4 declines smoothly from the start,
      it could be something else.

Stage boundaries for curriculum (80k / 6 stages = 13333 steps each):
  stage 0 k=1:   0 – 13332
  stage 1 k=2:  13333 – 26665
  stage 2 k=3:  26666 – 39998
  stage 3 k=4:  39999 – 53331   ← step-40k and step-50k are INSIDE this stage
  stage 4 k=5:  53332 – 66664   ← step-60k is inside this stage
  stage 5 k=6:  66665 – 79999   ← step-70k and step-80k are inside this stage

So the curriculum's k=4 performance should be PEAK around step-50k (end of stage 3)
and should start degrading at step-60k (training has switched to k=5).
"""
from __future__ import annotations
import sys, os, json, numpy as np, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from run_benchmark_eval import build_model, predict_proba
from run_tabarena_eval import load_task, score

TASKS_JSON = os.path.join(HERE, "tabarena_tasks.json")
CKPT = os.path.join(REPRO, "ckpt")
OUT = os.path.join(REPRO, "ouro_frozen_run.out")          # reuse output path
RESULTS_OUT = os.path.join(REPRO, "trend_diag_results.json")
GBDT_CACHE  = os.path.join(REPRO, "trend_results.json")

STEPS = [40000, 50000, 60000, 70000, 80000]

# __APPEND_MARKER__


def ck(run, step):
    return os.path.join(CKPT, run, f"step-{step}.ckpt")


def build_configs():
    """Return (tag, ckpt_path, nlayers, loop_k, run_name, step).
    Each (run, step, loop_k) triple is one model instance."""
    cfgs = []
    # loopk4: native k=4 at every step, PLUS k=1,2,3 at every step (cross-k evolution)
    for step in STEPS:
        for k in (1, 2, 3, 4):
            cfgs.append((f"loopk4_s{step//1000}k_k{k}",
                         ck("loopk4_12L_80000", step), 12, k, "loopk4", step))
    # loopk6: native k=6 at every step, PLUS k=4 for comparison
    for step in STEPS:
        for k in (4, 6):
            cfgs.append((f"loopk6_s{step//1000}k_k{k}",
                         ck("loopk6_12L_80000", step), 12, k, "loopk6", step))
    # curriculum: k=1..6 at every step — full cross-k trajectory
    for step in STEPS:
        for k in (1, 2, 3, 4, 5, 6):
            cfgs.append((f"curric_s{step//1000}k_k{k}",
                         ck("curric_k1to6_12L_80000", step), 12, k, "curric", step))
    # skip missing (shouldn't happen — all 8 steps confirmed present)
    out = [c for c in cfgs if os.path.exists(c[1])]
    miss = [c[0] for c in cfgs if not os.path.exists(c[1])]
    if miss:
        print(f"[skip missing] {miss}", flush=True)
    return out


def main():
    tasks = json.load(open(TASKS_JSON))
    clf = sorted([r for r in tasks if 2 <= r["cls"] <= 10], key=lambda r: r["rows"])
    print(f"datasets: {len(clf)}", flush=True)

    configs = build_configs()
    print(f"building {len(configs)} model instances...", flush=True)
    t0 = time.time()
    models = {}
    for tag, path, nl, k, run, step in configs:
        models[tag] = build_model(path, nl, k)
    print(f"built {len(models)} models in {time.time()-t0:.0f}s", flush=True)

    # load pre-computed GBDT
    gbdt = {}
    if os.path.exists(GBDT_CACHE):
        raw = json.load(open(GBDT_CACHE))["results"]
        gbdt = {g: raw[g] for g in ("catboost", "xgboost") if g in raw}
        print(f"loaded GBDT baselines ({len(next(iter(gbdt.values())))} datasets)", flush=True)

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
        print(f"  {name[:36]:36s} n={len(Xtr)} c={ncls}", flush=True)

    json.dump({"configs": [(t,p,nl,k,run,step) for t,p,nl,k,run,step in configs],
               "results": {**results, **gbdt}}, open(RESULTS_OUT, "w"), indent=1)
    summarize(results, gbdt, clf, configs)


def _mean(results, tag, names):
    v = [results[tag][n][1] for n in names
         if n in results.get(tag, {}) and not np.isnan(results[tag][n][1])]
    return np.mean(v) if v else float("nan")


def summarize(results, gbdt, clf, configs):
    names = [r["name"] for r in clf if r["name"] in next(iter(results.values()), {})]
    cat = _mean(gbdt, "catboost", names)

    print(f"\n{'='*68}", flush=True)
    print(f"TabArena ROC-AUC trajectory (baselines: catboost={cat:.4f})", flush=True)
    print(f"{'='*68}", flush=True)

    # (1) Fixed-k native trend — overfitting check
    print("\n--- (1) OVERFITTING CHECK: native-k AUC vs training step ---")
    print(f"{'step':>6}  {'loopk4@k4':>10}  {'loopk6@k6':>10}")
    for step in STEPS:
        k4 = _mean(results, f"loopk4_s{step//1000}k_k4", names)
        k6 = _mean(results, f"loopk6_s{step//1000}k_k6", names)
        print(f"{step//1000:>5}k  {k4:>10.4f}  {k6:>10.4f}")

    # (2) Curriculum cross-k trajectory — forgetting check
    print("\n--- (2) CURRICULUM CROSS-K TRAJECTORY (forgetting check) ---")
    print("Stage boundaries: 40k/50k in k=4 stage | 60k in k=5 stage | 70k/80k in k=6 stage")
    header = f"{'step':>6}  " + "  ".join(f"{'@k='+str(k):>8}" for k in (1,2,3,4,5,6))
    print(header)
    for step in STEPS:
        stage_note = {40000:"(k4-stage)", 50000:"(k4-stage)",
                      60000:"(k5-stage)", 70000:"(k6-stage)", 80000:"(k6-stage)"}
        cells = [_mean(results, f"curric_s{step//1000}k_k{k}", names) for k in (1,2,3,4,5,6)]
        row = f"{step//1000:>5}k  " + "  ".join(f"{v:>8.4f}" if not np.isnan(v) else f"{'---':>8}" for v in cells)
        print(f"{row}  {stage_note.get(step,'')}")

    # (3) loopk4 cross-k trajectory
    print("\n--- (3) LOOPK4 CROSS-K TRAJECTORY ---")
    header2 = f"{'step':>6}  " + "  ".join(f"{'@k='+str(k):>8}" for k in (1,2,3,4))
    print(header2)
    for step in STEPS:
        cells = [_mean(results, f"loopk4_s{step//1000}k_k{k}", names) for k in (1,2,3,4)]
        print(f"{step//1000:>5}k  " + "  ".join(f"{v:>8.4f}" for v in cells))

    # (4) loopk6: k=4 vs k=6 over steps
    print("\n--- (4) LOOPK6: k=4 vs k=6 AUC over steps ---")
    print(f"{'step':>6}  {'@k=4':>8}  {'@k=6':>8}")
    for step in STEPS:
        v4 = _mean(results, f"loopk6_s{step//1000}k_k4", names)
        v6 = _mean(results, f"loopk6_s{step//1000}k_k6", names)
        print(f"{step//1000:>5}k  {v4:>8.4f}  {v6:>8.4f}")


if __name__ == "__main__":
    main()
