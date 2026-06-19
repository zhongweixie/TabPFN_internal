#!/usr/bin/env python
"""Plan B — Phase 0 ceiling gate: ADAPTIVE ENSEMBLE DEPTH.

The decisive, cheap go/no-go for "FlashMemory's actual selling point on TabPFN":
spend the SAME average compute (n_estimators) but reallocate it across test rows —
hard rows get more ensemble members, easy rows get fewer. FlashMemory's win is a
Pareto improvement (equal compute, better accuracy / equal accuracy, less compute),
NOT raw accuracy. So the right ceiling question is:

  At a FIXED average budget B (mean estimators-per-row), can an ORACLE that knows the
  true label and allocates members optimally across rows BEAT uniform allocation?

  - If oracle-adaptive >> uniform on the Pareto curve  -> headroom EXISTS -> GO train a
    learned difficulty head (Phase 1).
  - If oracle-adaptive ~= uniform                      -> the per-member predictions have
    no row-specific convergence structure budgeting can exploit; adaptive depth is just
    variance reallocation with no net gain -> NO-GO, cheaply, before training anything.

Mechanism (one fit, no extra cost): predict_raw_logits returns per-ENSEMBLE-MEMBER
logits (K_MAX, M, C). The "n_estimators=k" prediction for a row is the running mean of
its first k members' softmax probs. So one K_MAX-member fit gives every budget level
k=1..K_MAX for free. The oracle uses the TRUE label ONLY to decide allocation — an
upper bound, not deployable (same legitimate "ceiling uses labels" pattern as the LOO
ceiling gate). Uniform is the deployable baseline.

Greedy marginal allocation: start every row at b=1, then repeatedly hand one extra
member to the row with the largest marginal loss reduction until the budget is spent.
(Greedy on possibly-non-monotone marginals is an approximate, *lower* bound on the true
oracle — so if greedy-oracle already beats uniform, the gap is real.)
"""

from __future__ import annotations

import heapq
import logging
import sys
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("adaptive_ceiling_results.log", mode="w")],
)
log = logging.getLogger("adaptive_ceiling")

DEVICE = "cuda"
MODEL_PATH = (
    "/home/zxiebk/workspace/model/tabpfn_2_5/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)
SEEDS = [0, 1]
N_TRAIN = 1000       # bigger N ok here: no LOO, just one K_MAX-member fit
N_TEST = 300
K_MAX = 32           # ensemble-member pool; every budget 1..K_MAX comes free from it
EPS = 1e-7
BUDGETS = [1, 2, 4, 8, 16]   # average-per-row budgets B to compare uniform vs adaptive

def mk(seed):
    return TabPFNClassifier(
        n_estimators=K_MAX, device=DEVICE, model_path=MODEL_PATH,
        random_state=seed, fit_mode="fit_preprocessors",
        ignore_pretraining_limits=True,
    )


def member_probs(clf, Xte):
    """One K_MAX-member fit -> per-member, class-aligned probs (K, M, C).

    predict_raw_logits returns de-permuted logits aligned to clf.classes_, so members
    are directly comparable. Softmax each member; the budget-k prediction for a row is
    the running mean of its first k members (averaging probabilities)."""
    logits = clf.predict_raw_logits(Xte)            # (K, M, C), numpy
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 4:                            # (K, M, 1, C) on some engines
        logits = logits[:, :, 0, :]
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)             # (K, M, C)


def running_means(p):
    """p:(K,M,C) -> cumulative-mean predictions cm:(K,M,C), cm[k]=mean of first k+1 members."""
    return np.cumsum(p, axis=0) / np.arange(1, p.shape[0] + 1)[:, None, None]


def row_loss(cm, col):
    """Per-row negative-log-likelihood of true class at each budget. cm:(K,M,C),
    col:(M,) true-class index. Returns L:(K,M), L[k,m] = loss of row m at budget k+1."""
    K, M, _ = cm.shape
    pt = cm[:, np.arange(M), col]                   # (K, M) prob of true class
    return -np.log(np.clip(pt, EPS, 1.0))


def row_correct(cm, col):
    """Per-row correctness (K,M) at each budget: argmax == true."""
    return (cm.argmax(-1) == col[None, :]).astype(np.float64)


def alloc_uniform(M, b):
    """Every row gets exactly b members (the deployable baseline)."""
    return np.full(M, b, dtype=int)


def alloc_oracle(loss, total):
    """Greedy oracle: start each row at b=1, repeatedly give one member to the row with
    the largest marginal true-loss reduction, until `total` members are spent.

    Greedy on possibly-non-monotone marginals is an approximate LOWER bound on the true
    optimum -> if greedy already beats uniform, the headroom is real. loss:(K,M)."""
    K, M = loss.shape
    b = np.ones(M, dtype=int)
    heap = []  # (-marginal_gain, row, current_budget)
    for m in range(M):
        if K > 1:
            heapq.heappush(heap, (-(loss[0, m] - loss[1, m]), m, 1))
    spent = M
    while spent < total and heap:
        neg_gain, m, cur = heapq.heappop(heap)
        if cur >= K:                                # row maxed out
            continue
        b[m] = cur + 1
        spent += 1
        if cur + 1 < K:
            nxt = loss[cur, m] - loss[cur + 1, m]
            heapq.heappush(heap, (-nxt, m, cur + 1))
    return b


def alloc_unsup(p, total):
    """Deployable proxy: allocate by member DISAGREEMENT (no labels). Difficulty =
    variance of the true-class-free predictive entropy across members; budget the hard
    rows more. Surrogate marginal gain = difficulty / (b*(b+1)) (variance-of-mean
    diminishing returns), greedily assigned. Previews what a learned head could do."""
    K, M, _ = p.shape
    # disagreement signal: mean across members of entropy + spread of member argmaxes
    ent = -(p * np.log(np.clip(p, EPS, 1.0))).sum(-1)        # (K, M)
    diff = ent.mean(0)                                       # (M,) avg predictive entropy
    diff = diff + p.std(0).sum(-1)                           # + per-class member spread
    diff = np.clip(diff, 1e-6, None)
    b = np.ones(M, dtype=int)
    heap = [(-diff[m] / (1 * 2), m, 1) for m in range(M)]
    heapq.heapify(heap)
    spent = M
    while spent < total and heap:
        _, m, cur = heapq.heappop(heap)
        if cur >= K:
            continue
        b[m] = cur + 1; spent += 1
        if cur + 1 < K:
            g = diff[m] / ((cur + 1) * (cur + 2))
            heapq.heappush(heap, (-g, m, cur + 1))
    return b


def gather(cm, b):
    """Predictions for chosen per-row budgets. cm:(K,M,C), b:(M,) in [1,K] -> (M,C)."""
    M = b.shape[0]
    return cm[b - 1, np.arange(M), :]


def evaluate_alloc(cm, b, yte_idx, labels_present):
    """acc + log-loss + auc for a given allocation."""
    pred = gather(cm, b)                            # (M, C)
    yhat = pred.argmax(1)
    acc = accuracy_score(yte_idx, yhat)
    ll = log_loss(yte_idx, pred, labels=list(range(pred.shape[1])))
    try:
        if pred.shape[1] == 2:
            auc = roc_auc_score(yte_idx, pred[:, 1])
        else:
            auc = roc_auc_score(yte_idx, pred, multi_class="ovr",
                                average="macro", labels=list(range(pred.shape[1])))
    except ValueError:
        auc = float("nan")
    return acc, ll, auc


def alloc_oracle_crossfit(p, col, total, rng):
    """Cross-fitted oracle: decide per-row budget on DECIDE members, evaluate on EVAL
    members (disjoint). Removes the 'pick a lucky k on the same members you score'
    snooping artifact. If the gain survives this, some rows GENUINELY converge slower
    (real adaptive-depth structure); if it vanishes, the plain oracle was cheating.

    Returns (acc_adapt, ll_adapt, avg_b, acc_unif, ll_unif) on EVAL members."""
    K, M, C = p.shape
    perm = rng.permutation(K)
    d_idx, e_idx = perm[:K // 2], perm[K // 2:]
    cm_d = running_means(p[d_idx])          # decide-set running means
    cm_e = running_means(p[e_idx])          # eval-set running means
    Kd = cm_d.shape[0]
    loss_d = row_loss(cm_d, col)            # decide budgets by DECIDE loss only
    b = alloc_oracle(loss_d, min(total, Kd * M))
    b = np.clip(b, 1, cm_e.shape[0])
    pred = cm_e[b - 1, np.arange(M), :]     # score on EVAL members
    acc = accuracy_score(col, pred.argmax(1))
    ll = log_loss(col, pred, labels=list(range(C)))
    bu = np.clip(alloc_uniform(M, int(round(b.mean()))), 1, cm_e.shape[0])
    pu = cm_e[bu - 1, np.arange(M), :]      # uniform on SAME eval members, same avg budget
    accu = accuracy_score(col, pu.argmax(1))
    llu = log_loss(col, pu, labels=list(range(C)))
    return acc, ll, float(b.mean()), accu, llu


def run_one(name, X, y, seed):
    rng = np.random.RandomState(seed)
    labels = np.unique(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    if len(Xtr) > N_TRAIN:
        i = rng.choice(len(Xtr), N_TRAIN, replace=False); Xtr, ytr = Xtr[i], ytr[i]
    if len(Xte) > N_TEST:
        i = rng.choice(len(Xte), N_TEST, replace=False); Xte, yte = Xte[i], yte[i]

    clf = mk(seed); clf.fit(Xtr, ytr)
    p = member_probs(clf, Xte)                      # (K, M, C)
    cls_index = {c: i for i, c in enumerate(clf.classes_)}
    col = np.array([cls_index[c] for c in yte])     # true-class idx aligned to p
    cm = running_means(p)
    loss = row_loss(cm, col)

    M = len(yte)
    out = {}
    for B in BUDGETS:
        total = B * M
        bu = alloc_uniform(M, B)
        bo = alloc_oracle(loss, total)
        bs = alloc_unsup(p, total)
        out[B] = {
            "uniform": evaluate_alloc(cm, bu, col, labels),
            "oracle":  evaluate_alloc(cm, bo, col, labels),
            "unsup":   evaluate_alloc(cm, bs, col, labels),
            "avg_b_oracle": float(bo.mean()),
            "avg_b_unsup": float(bs.mean()),
            # cross-fitted oracle: snoop-free Δ on held-out members
            "xfit": alloc_oracle_crossfit(p, col, total, np.random.RandomState(seed + 7)),
        }
    return out


def aggregate(name, runs):
    log.info("=" * 78)
    log.info("DATASET %s  (n_seeds=%d, K_MAX=%d)", name, len(runs), K_MAX)
    log.info("  budget |  uniform acc/ll/auc  |  ORACLE (Δacc Δll)   |  unsup (Δacc Δll)"
             "  |  XFIT-oracle Δacc Δll (snoop-free)")
    for B in BUDGETS:
        u = np.array([[*r[B]["uniform"]] for r in runs]).mean(0)
        o = np.array([[*r[B]["oracle"]] for r in runs]).mean(0)
        s = np.array([[*r[B]["unsup"]] for r in runs]).mean(0)
        ab_o = np.mean([r[B]["avg_b_oracle"] for r in runs])
        ab_s = np.mean([r[B]["avg_b_unsup"] for r in runs])
        # xfit: (acc_adapt, ll_adapt, avg_b, acc_unif, ll_unif) -> Δ vs its own uniform
        xf = np.array([[*r[B]["xfit"]] for r in runs]).mean(0)
        xdacc, xdll = xf[0] - xf[3], xf[1] - xf[4]
        log.info("  B=%2d   |  %.3f %.3f %.3f  |  %.3f %.3f (%+.3f %+.3f)  |  "
                 "%.3f %.3f (%+.3f %+.3f) [b̄ o=%.1f s=%.1f]  |  %+.3f %+.3f (b̄=%.1f)",
                 B, u[0], u[1], u[2],
                 o[0], o[1], o[0] - u[0], o[1] - u[1],
                 s[0], s[1], s[0] - u[0], s[1] - u[1], ab_o, ab_s,
                 xdacc, xdll, xf[2])
    log.info("  READ: ORACLE Δ = optimistic ceiling (snoops). XFIT Δ = snoop-free "
             "(decide budget on half the members, score on the other half). "
             "If XFIT Δ collapses to ~0, the oracle was cherry-picking k, not finding "
             "slow-converging rows -> NO-GO. unsup = deployable label-free proxy.")


def load_datasets():
    out = [("breast_cancer", *load_breast_cancer(return_X_y=True))]
    for nm in ["phoneme", "electricity", "qsar-biodeg"]:
        try:
            d = fetch_openml(nm, version=1, as_frame=False, parser="liac-arff")
            X = d.data.astype("float32")
            mask = np.isfinite(X).all(1)
            X = X[mask]
            cls = np.unique(d.target)
            y = np.searchsorted(cls, d.target).astype(int)[mask]
            out.append((nm, X, y))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", nm, str(e)[:80])
    return out


def main():
    log.info("ADAPTIVE-DEPTH ceiling gate | N_tr=%d N_te=%d K_MAX=%d budgets=%s seeds=%s",
             N_TRAIN, N_TEST, K_MAX, BUDGETS, SEEDS)
    for name, X, y in load_datasets():
        t0 = time.time()
        runs = [run_one(name, X, y, s) for s in SEEDS]
        aggregate(name, runs)
        log.info("  (%.1fs)", time.time() - t0)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()
