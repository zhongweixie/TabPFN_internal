#!/usr/bin/env python
"""Full-training-set global indexer: many LOO query samples per dataset.

Answers the open question after the 14-dataset scale-up NO-GO (win-rate 7%): that
run used only 80 query rows/dataset. The method is sample-sensitive (60->80 query
lifted phoneme 0.466->0.620), so this maximizes legitimate query supervision.

Per dataset (all from non-test data, disjoint):
  context (N_CTX) : in-context train set for TabPFN
  query   (N_QUERY, LARGE) : training-set-LOO golden labels; each query row is NOT in
                             its own context, label known -> no test leak.
  test    (N_TEST) : held-out eval, never used for labels.
One indexer trained on pooled (query_repr, ctx_repr, golden) over all datasets;
held-out test eval. Win-rate vs full / knn.

vs run_indexer_global: N_QUERY 80 -> 240 (3x the supervision), same leak-free protocol.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer

sys.path.insert(0, ".")
import run_indexer_global as GL  # noqa: E402
import run_indexer_gate as G  # noqa: E402
from run_loo_ceiling import loo_influence  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("indexer_fulltrain_results.log", mode="w")],
)
log = logging.getLogger("fulltrain")

# Larger context + MUCH larger query set (the point of this experiment)
GL.N_CTX = 150
GL.N_QUERY = 240          # 3x the scale-up's 80
GL.N_TEST = 60
KEEP_FRAC = GL.KEEP_FRAC

DATASETS = [
    "phoneme", "electricity", "qsar-biodeg", "diabetes",
    "blood-transfusion-service-center", "banknote-authentication", "kc1", "pc1",
    "wdbc", "ilpd", "steel-plates-fault", "ozone-level-8hr",
]  # + breast_cancer = 13 (need >= N_CTX+N_QUERY+N_TEST = 450 rows)
SEEDS = [0, 1]

# __APPEND_MARKER__


def make_samples_big(name, X, y, seed):
    """Like GL.make_samples but with explicit sizes (context+LARGE query+test), and
    it adapts the query size down if the dataset is small. Leak-free."""
    from sklearn.model_selection import train_test_split
    import torch
    labels = np.unique(y)
    n = len(y)
    # reserve test (N_TEST) + context (N_CTX); the rest (capped at N_QUERY) is query
    n_test = GL.N_TEST
    Xrest, Xte, yrest, yte = train_test_split(X, y, test_size=n_test / n, random_state=seed, stratify=y)
    n_ctx = min(GL.N_CTX, len(yrest) // 2)
    Xc, Xq, yc, yq = train_test_split(Xrest, yrest, train_size=n_ctx / len(yrest),
                                      random_state=seed, stratify=yrest)
    Xc, yc = Xc[:n_ctx], yc[:n_ctx]
    Xq, yq = Xq[:GL.N_QUERY], yq[:GL.N_QUERY]      # large query set
    Xte, yte = Xte[:n_test], yte[:n_test]

    clf = G.mk_clf(seed); clf.fit(Xc, yc)
    infl, _ = loo_influence(Xc, yc, Xq, yq, seed)
    Nc = len(Xc); keep = max(1, int(round(Nc * KEEP_FRAC)))
    golden = np.zeros((len(Xq), Nc), dtype=np.float32)
    for m in range(len(Xq)):
        golden[m, np.argsort(-infl[m])[:keep]] = 1.0
    dq = G.extract_layer_reprs(clf, Xq)
    ctx_repr = dq[:Nc].float(); query_repr = dq[Nc:].float()
    dte = G.extract_layer_reprs(clf, Xte)
    ctx_repr_te = dte[:Nc].float(); test_repr = dte[Nc:].float()
    return dict(name=name, clf=clf, labels=labels, Xc=Xc, yc=yc, Xte=Xte, yte=yte,
                ctx_repr=ctx_repr, query_repr=query_repr, golden=golden,
                ctx_repr_te=ctx_repr_te, test_repr=test_repr, nq=len(Xq))


def load_all():
    raw = [("breast_cancer", *load_breast_cancer(return_X_y=True))]
    for nm in DATASETS:
        try:
            d = fetch_openml(nm, version=1, as_frame=False, parser="liac-arff")
            X = np.asarray(d.data, dtype="float32")
            if np.isnan(X).any():
                c = np.nanmean(X, axis=0); ii = np.where(np.isnan(X)); X[ii] = np.take(c, ii[1])
            cls = np.unique(d.target); y = np.searchsorted(cls, d.target).astype(int)
            raw.append((nm, X, y))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", nm, str(e)[:60])
    return raw


def main():
    log.info("FULL-TRAIN global indexer | ctx=%d query<=%d test=%d | seeds=%s | keep=%.0f%%",
             GL.N_CTX, GL.N_QUERY, GL.N_TEST, SEEDS, KEEP_FRAC * 100)
    raw = load_all()
    per_ds = {nm: [] for nm, _, _ in raw}
    for seed in SEEDS:
        t0 = time.time()
        samples = []
        for nm, X, y in raw:
            try:
                samples.append(make_samples_big(nm, X, y, seed))
            except Exception as e:  # noqa: BLE001
                log.warning("seed%d skip %s: %s", seed, nm, str(e)[:50])
        log.info("seed%d built %d samples (query sizes: %s) in %.0fs", seed, len(samples),
                 [s["nq"] for s in samples], time.time() - t0)
        idx = GL.train_global_indexer(samples)
        for s in samples:
            clf, labels = s["clf"], s["labels"]; Xte, yte = s["Xte"], s["yte"]
            full = G._score(yte, clf.predict_proba(Xte), labels)[1]
            sc = idx.scores(s["test_repr"], s["ctx_repr_te"])
            ixv = G._score(yte, G.predict_with_indexer(clf, Xte, sc), labels)[1]
            knn = G._score(yte, G.knn_topk_proba(clf, s["Xc"], s["yc"], Xte, KEEP_FRAC, labels), labels)[1]
            per_ds[s["name"]].append((full, ixv, knn))
            log.info("seed%d %-26s full=%.3f indexer=%.3f knn=%.3f bf=%s bk=%s",
                     seed, s["name"], full, ixv, knn, ixv > full, ixv > knn)
    log.info("=" * 66)
    rows = [(nm, np.array(v)) for nm, v in per_ds.items() if v]
    bf = sum((v[:, 1] > v[:, 0]).sum() for _, v in rows)
    bk = sum((v[:, 1] > v[:, 2]).sum() for _, v in rows)
    tot = sum(len(v) for _, v in rows)
    for nm, v in rows:
        log.info("%-26s indexer-full=%+.3f indexer-knn=%+.3f", nm,
                 (v[:, 1] - v[:, 0]).mean(), (v[:, 1] - v[:, 2]).mean())
    log.info("WIN-RATE: indexer>full %d/%d (%.0f%%)  indexer>knn %d/%d (%.0f%%)",
             bf, tot, 100 * bf / tot, bk, tot, 100 * bk / tot)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()

