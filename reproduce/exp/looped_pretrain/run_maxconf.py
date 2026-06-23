#!/usr/bin/env python
"""maxConf: zero-training adaptive loop_k baseline.

Uses the c2 (trained @ loop_k=2) weights, runs inference at loop_k=1..KMAX, and for
each test row picks the prediction with the highest confidence (max class prob) — a
LABEL-FREE per-row selection rule. This is the cheapest possible "adaptive depth":
no training, just pick the loop count whose output the model is most confident about.

Compared on the full TabArena 38-dataset classification suite (ROC-AUC OvR) against:
  - fixed k=1, fixed k=2 (c2's trained depth), fixed k=3, fixed k=4   [single-depth baselines]
  - maxConf   (per-row argmax-confidence over k=1..KMAX)
  - mean-ens  (per-row mean of probs over k=1..KMAX)                  [no selection]
  - label-oracle (per-row pick k with highest TRUE-class prob)        [cheating UB]

Probe (6 datasets, true-prob) showed maxConf consistently beats fixed-k2 by +0.01..0.03
and captures ~30-40% of the oracle headroom — UNLIKE the earlier adaptive-depth NO-GO
where the deployable signal was ~0. This script checks whether that survives on the
full suite under the proper ROC-AUC metric (true-prob gains don't fully convert to AUC).

CAVEAT (honest): c2's weights are trained ONLY at k=2; k=1/3/4 are out-of-training-
distribution inference. maxConf may be winning largely by AVOIDING the degenerate
off-k outputs. A genuinely variable-k-trained model is the real test (future work).
"""

from __future__ import annotations

import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, HERE)

from sklearn.metrics import roc_auc_score
import run_tabarena_eval as T   # reuse load_task, score, REPO, TASKS_JSON
import looped_step2 as L
from run_benchmark_eval import build_model, predict_proba

KMAX = 4
KS = list(range(1, KMAX + 1))

# __APPEND_MARKER__


def renorm(p):
    return p / p.sum(1, keepdims=True).clip(min=1e-12)


def auc(yte, proba, ncls):
    proba = renorm(proba)
    try:
        if ncls == 2:
            return roc_auc_score(yte, proba[:, 1])
        return roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                             labels=list(range(proba.shape[1])))
    except ValueError:
        return float("nan")


def main():
    tasks = json.load(open(T.TASKS_JSON))
    clf = sorted([r for r in tasks if 2 <= r["cls"] <= 10], key=lambda r: r["rows"])
    print(f"TabArena clf datasets: {len(clf)} | KMAX={KMAX}", flush=True)

    c2 = build_model(f"{T.REPO}/ckpt_c2_loop/step-80000.ckpt", 12, 2)

    methods = [f"fix_k{k}" for k in KS] + ["maxConf", "mean_ens", "label_oracle"]
    res = {m: {} for m in methods}

    for r in clf:
        name, tid = r["name"], r["task"]
        try:
            Xtr, ytr, Xte, yte = T.load_task(tid)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {name}: {str(e)[:50]}", flush=True); continue
        ncls = int(max(ytr.max(), yte.max())) + 1
        M = len(yte)
        # per-k probabilities (M, C), renormalized
        P = {}
        for k in KS:
            L.set_loop_on_model(c2, k)
            P[k] = renorm(predict_proba(c2, Xtr, ytr, Xte))
        # fixed-k
        for k in KS:
            res[f"fix_k{k}"][name] = auc(yte, P[k], ncls)
        # maxConf: per row pick the k whose prediction has highest max-prob (confidence)
        conf = np.stack([P[k].max(1) for k in KS], 1)        # (M, K)
        bestk = conf.argmax(1)                                # index into KS
        sel = np.stack([P[k] for k in KS], 1)[np.arange(M), bestk]  # (M, C)
        res["maxConf"][name] = auc(yte, sel, ncls)
        # mean ensemble (no selection)
        res["mean_ens"][name] = auc(yte, np.mean([P[k] for k in KS], 0), ncls)
        # label oracle: per row pick k with highest TRUE-class prob (cheating UB)
        tp = np.stack([P[k][np.arange(M), yte] for k in KS], 1)  # (M, K)
        ok = tp.argmax(1)
        orc = np.stack([P[k] for k in KS], 1)[np.arange(M), ok]
        res["label_oracle"][name] = auc(yte, orc, ncls)
        print(f"  {name[:30]:30s} k2={res['fix_k2'][name]:.3f} "
              f"maxConf={res['maxConf'][name]:.3f} oracle={res['label_oracle'][name]:.3f}",
              flush=True)

    json.dump(res, open(os.path.join(HERE, "maxconf_results.json"), "w"), indent=1)
    summarize(res)


def summarize(res):
    names = sorted(res["fix_k2"].keys())
    methods = list(res.keys())
    print(f"\n{'='*64}\nSUMMARY mean ROC-AUC over {len(names)} datasets\n{'='*64}")
    means = {}
    for m in methods:
        vals = [res[m][n] for n in names if not np.isnan(res[m][n])]
        means[m] = np.mean(vals)
        print(f"  {m:14s} {means[m]:.4f}  (n={len(vals)})")
    # maxConf vs fix_k2 win/tie/loss + per-dataset deltas
    print(f"\nmaxConf vs fix_k2 (the key comparison):")
    w = t = l = 0; deltas = []
    for n in names:
        a, b = res["maxConf"][n], res["fix_k2"][n]
        if np.isnan(a) or np.isnan(b): continue
        deltas.append(a - b)
        if a > b + 1e-9: w += 1
        elif abs(a - b) <= 1e-9: t += 1
        else: l += 1
    print(f"  W={w} T={t} L={l} | win+tie={100*(w+t)/(w+t+l):.0f}% | "
          f"mean Δ={np.mean(deltas):+.4f} (median {np.median(deltas):+.4f})")
    # how much of the oracle gap does maxConf capture?
    gap_oracle = means["label_oracle"] - means["fix_k2"]
    gap_maxconf = means["maxConf"] - means["fix_k2"]
    if gap_oracle > 0:
        print(f"  maxConf captures {100*gap_maxconf/gap_oracle:.0f}% of the (cheating) oracle headroom "
              f"[maxConf +{gap_maxconf:.4f} / oracle +{gap_oracle:.4f}]")


if __name__ == "__main__":
    main()
