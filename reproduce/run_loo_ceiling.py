#!/usr/bin/env python
"""Phase 0 ceiling gate: does a TRUE per-query training-row subset beat `full`?

The decisive go/no-go for the learned-indexer direction. Earlier probes used the
model's own attention (a heuristic); here we use leave-one-out (LOO) INFLUENCE — the
behavioural ground truth of which training rows help each test row's prediction.

Why naive refit (not cache slicing): a subagent claimed KV-cache per-row structure
makes LOO cheap by slicing row j, but train-row cached keys are computed by attending
over ALL train rows (then 24 layers of mixing), so dropping row j's cache entry is NOT
the same as row j never existing (verified: cache-slice vs refit maxdiff 0.37). So we
refit on X\{j}. Feasible only at small N.

Protocol per (dataset, seed, noise-setting):
  1. fit full -> baseline p_full(true) per test row.
  2. LOO: for each train row j, refit on X\{j}, record influence
     I[m,j] = p_full(true_m) - p_{-j}(true_m). Positive => row j helped test m.
  3. For each keep-fraction k, select per test row its top-k% rows by I[m,:],
     predict that test row with ONLY its subset, measure ROC-AUC + accuracy.
  4. Compare to full. Run CLEAN and LABEL-FLIP-50% noise.

CEILING question: even with the ground-truth-best subset, can sparse beat/hold full?
  - clean: if LOO-top-k can't reach full -> importance not concentrated -> no headroom.
  - noise: if LOO-top-k recovers accuracy by excluding flipped rows (full collapses
    ~0.92->0.29) -> denoising headroom EXISTS -> GO for a learned indexer. Else no-go.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("loo_ceiling_results.log", mode="w"),
    ],
)
log = logging.getLogger("loo_ceiling")

DEVICE = "cuda"
MODEL_PATH = (
    "/home/zxiebk/workspace/model/tabpfn_2_5/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)
SEEDS = [0, 1]
N_TRAIN = 120        # small enough for naive LOO (N refits per test batch)
N_TEST = 40
KEEP_FRACS = [0.10, 0.25, 0.50, 0.75, 0.90]
NOISE_FLIP = 0.50    # fraction of train labels flipped in the noise setting

# __APPEND_MARKER__


def mk(seed):
    return TabPFNClassifier(
        n_estimators=1, device=DEVICE, model_path=MODEL_PATH,
        random_state=seed, fit_mode="fit_preprocessors",
        ignore_pretraining_limits=True,
    )


def flip_labels(y, labels, frac, rng):
    """Flip `frac` of labels to a different class (on-manifold hard noise)."""
    y = y.copy()
    m = int(len(y) * frac)
    idx = rng.choice(len(y), m, replace=False)
    for i in idx:
        others = [c for c in labels if c != y[i]]
        y[i] = rng.choice(others)
    return y


def loo_influence(Xtr, ytr, Xte, yte, seed):
    """Return influence matrix I[M,N]: I[m,j] = p_full(true_m) - p_{-j}(true_m)."""
    labels = np.unique(ytr)
    clf = mk(seed); clf.fit(Xtr, ytr)
    p_full = clf.predict_proba(Xte)
    cls_index = {c: i for i, c in enumerate(clf.classes_)}
    col = np.array([cls_index[c] for c in yte])
    base = p_full[np.arange(len(yte)), col]
    N = len(ytr)
    infl = np.zeros((len(yte), N))
    for j in range(N):
        keep = np.ones(N, bool); keep[j] = False
        # subset must still contain all classes, else skip (leave influence 0)
        if len(np.unique(ytr[keep])) < len(labels):
            continue
        c = mk(seed); c.fit(Xtr[keep], ytr[keep])
        pj = c.predict_proba(Xte)
        ci = {cc: i for i, cc in enumerate(c.classes_)}
        colj = np.array([ci[cc] for cc in yte])
        infl[:, j] = base - pj[np.arange(len(yte)), colj]
    return infl, p_full


def eval_per_query_topk(Xtr, ytr, Xte, yte, infl, keep_frac, seed, labels):
    """Each test row predicted with ITS OWN top-k influential train rows.

    Groups test rows by identical selected-index sets so we refit once per distinct
    subset instead of once per test row (big speedup when keep is large / rows overlap).
    Class-coverage failures fall back to the full set immediately (no escalation loop).
    """
    N = len(ytr)
    keep = max(int(round(N * keep_frac)), len(labels) * 2)
    # build each test row's selected index set
    sel = [tuple(sorted(np.argsort(-infl[m])[:keep].tolist())) for m in range(len(Xte))]
    proba = np.zeros((len(Xte), len(labels)))
    # cache fitted models per distinct subset
    cache = {}
    for m, key in enumerate(sel):
        idx = np.array(key)
        if len(np.unique(ytr[idx])) < len(labels):
            idx = np.arange(N)          # fallback: full set (no class dropped)
            key = ("FULL",)
        if key not in cache:
            c = mk(seed); c.fit(Xtr[idx], ytr[idx])
            cache[key] = c
        c = cache[key]
        pm = c.predict_proba(Xte[m:m + 1])
        for i, cc in enumerate(c.classes_):
            proba[m, list(labels).index(cc)] = pm[0, i]
    return proba


def score(y_true, proba, labels):
    pred = labels[np.argmax(proba, axis=1)]
    acc = accuracy_score(y_true, pred)
    try:
        auc = (roc_auc_score(y_true, proba[:, 1]) if len(labels) == 2
               else roc_auc_score(y_true, proba, multi_class="ovr",
                                  average="macro", labels=labels))
    except ValueError:
        auc = float("nan")
    return acc, auc


def full_proba_aligned(Xtr, ytr, Xte, seed, labels):
    c = mk(seed); c.fit(Xtr, ytr)
    p = c.predict_proba(Xte)
    out = np.zeros((len(Xte), len(labels)))
    for i, cc in enumerate(c.classes_):
        out[:, list(labels).index(cc)] = p[:, i]
    return out


def run_one(name, X, y, seed, noise):
    rng = np.random.RandomState(seed)
    labels = np.unique(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    # cap sizes for naive LOO feasibility
    if len(Xtr) > N_TRAIN:
        i = rng.choice(len(Xtr), N_TRAIN, replace=False); Xtr, ytr = Xtr[i], ytr[i]
    if len(Xte) > N_TEST:
        i = rng.choice(len(Xte), N_TEST, replace=False); Xte, yte = Xte[i], yte[i]
    if noise:
        ytr = flip_labels(ytr, labels, NOISE_FLIP, rng)

    full_acc, full_auc = score(yte, full_proba_aligned(Xtr, ytr, Xte, seed, labels), labels)
    infl, _ = loo_influence(Xtr, ytr, Xte, yte, seed)
    frac_neg = float((infl < 0).mean())
    res = {"full": (full_acc, full_auc), "frac_neg": frac_neg}
    for kf in KEEP_FRACS:
        pr = eval_per_query_topk(Xtr, ytr, Xte, yte, infl, kf, seed, labels)
        res[kf] = score(yte, pr, labels)
    return res


def aggregate(name, runs, noise):
    log.info("##### %s  noise=%s  (n_seeds=%d) #####", name,
             "flip50" if noise else "clean", len(runs))
    fa = np.array([r["full"][0] for r in runs]); fu = np.array([r["full"][1] for r in runs])
    log.info("  full        acc=%.3f±%.3f  auc=%.3f±%.3f  | frac_neg_influence=%.2f",
             fa.mean(), fa.std(), fu.mean(), fu.std(),
             np.mean([r["frac_neg"] for r in runs]))
    for kf in KEEP_FRACS:
        a = np.array([r[kf][0] for r in runs]); u = np.array([r[kf][1] for r in runs])
        log.info("  LOO k=%3.0f%%  acc=%.3f±%.3f (%+.3f)  auc=%.3f±%.3f (%+.3f)",
                 kf * 100, a.mean(), a.std(), a.mean() - fa.mean(),
                 u.mean(), u.std(), u.mean() - fu.mean())


def load_datasets():
    out = [("breast_cancer", *load_breast_cancer(return_X_y=True))]
    for nm in ["phoneme", "electricity"]:
        try:
            d = fetch_openml(nm, version=1, as_frame=False, parser="liac-arff")
            X = d.data.astype("float32")
            cls = np.unique(d.target); y = np.searchsorted(cls, d.target).astype(int)
            out.append((nm, X, y))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", nm, str(e)[:80])
    return out


def main():
    log.info("LOO ceiling gate | N_train=%d N_test=%d | keep=%s | seeds=%s | flip=%.0f%%",
             N_TRAIN, N_TEST, KEEP_FRACS, SEEDS, NOISE_FLIP * 100)
    for name, X, y in load_datasets():
        log.info("=" * 72)
        log.info("DATASET %s  shape=%s  classes=%d", name, X.shape, len(np.unique(y)))
        for noise in (False, True):
            t0 = time.time()
            runs = [run_one(name, X, y, s, noise) for s in SEEDS]
            aggregate(name, runs, noise)
            log.info("  (%.1fs)", time.time() - t0)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()


