#!/usr/bin/env python
"""Scale-up of the global cross-dataset indexer: 14 datasets x multi-seed.

Question: does the electricity-style success (held-out indexer > full AND > knn)
become common as the number of training datasets grows, or stay rare? One indexer
trained on the pooled query samples of ALL datasets; each dataset evaluated on its
own held-out test rows (never used for labels). Leak-free (same protocol as
run_indexer_global.make_samples).
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
from run_loo_ceiling import eval_per_query_topk, loo_influence  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("indexer_scaleup_results.log", mode="w")],
)
log = logging.getLogger("scaleup")

DATASETS = [
    "phoneme", "electricity", "qsar-biodeg", "diabetes",
    "blood-transfusion-service-center", "banknote-authentication", "kc1", "pc1",
    "wdbc", "ilpd", "climate-model-simulation-crashes", "steel-plates-fault",
    "ozone-level-8hr",
]  # + breast_cancer (sklearn) = 14
SEEDS = [0, 1]
KEEP_FRAC = GL.KEEP_FRAC

# __APPEND_MARKER__


def load_all():
    raw = [("breast_cancer", *load_breast_cancer(return_X_y=True))]
    for nm in DATASETS:
        try:
            d = fetch_openml(nm, version=1, as_frame=False, parser="liac-arff")
            X = np.asarray(d.data, dtype="float32")
            # impute NaNs with column means (some OpenML sets have missing values)
            if np.isnan(X).any():
                col = np.nanmean(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col, inds[1])
            cls = np.unique(d.target)
            y = np.searchsorted(cls, d.target).astype(int)
            raw.append((nm, X, y))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", nm, str(e)[:60])
    return raw


def eval_one(idx, s):
    clf, labels = s["clf"], s["labels"]
    Xte, yte = s["Xte"], s["yte"]
    full = G._score(yte, clf.predict_proba(Xte), labels)[1]
    sc = idx.scores(s["test_repr"], s["ctx_repr_te"])
    ixv = G._score(yte, G.predict_with_indexer(clf, Xte, sc), labels)[1]
    knn = G._score(yte, G.knn_topk_proba(clf, s["Xc"], s["yc"], Xte, KEEP_FRAC, labels), labels)[1]
    return full, ixv, knn


def main():
    log.info("SCALE-UP global indexer | %d datasets | seeds=%s | ctx=%d query=%d test=%d",
             len(DATASETS) + 1, SEEDS, GL.N_CTX, GL.N_QUERY, GL.N_TEST)
    raw = load_all()
    log.info("loaded %d datasets", len(raw))
    per_ds = {nm: [] for nm, _, _ in raw}     # nm -> list of (full,idx,knn) across seeds
    for seed in SEEDS:
        t0 = time.time()
        samples = []
        for nm, X, y in raw:
            try:
                samples.append(GL.make_samples(nm, X, y, seed))
            except Exception as e:  # noqa: BLE001
                log.warning("seed%d skip %s: %s", seed, nm, str(e)[:50])
        idx = GL.train_global_indexer(samples)
        for s in samples:
            full, ixv, knn = eval_one(idx, s)
            per_ds[s["name"]].append((full, ixv, knn))
            log.info("seed%d %-26s full=%.3f indexer=%.3f knn=%.3f beats_full=%s beats_knn=%s",
                     seed, s["name"], full, ixv, knn, ixv > full, ixv > knn)
        log.info("seed %d done in %.1fs", seed, time.time() - t0)

    # aggregate: how often does the indexer beat full / knn across datasets+seeds?
    log.info("=" * 66)
    rows = [(nm, np.array(v)) for nm, v in per_ds.items() if v]
    bf = sum((v[:, 1] > v[:, 0]).sum() for _, v in rows)
    bk = sum((v[:, 1] > v[:, 2]).sum() for _, v in rows)
    tot = sum(len(v) for _, v in rows)
    for nm, v in rows:
        log.info("%-26s indexer-full=%+.3f indexer-knn=%+.3f (mean over %d seeds)",
                 nm, (v[:, 1] - v[:, 0]).mean(), (v[:, 1] - v[:, 2]).mean(), len(v))
    log.info("WIN-RATE: indexer>full %d/%d (%.0f%%)  indexer>knn %d/%d (%.0f%%)",
             bf, tot, 100 * bf / tot, bk, tot, 100 * bk / tot)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()

