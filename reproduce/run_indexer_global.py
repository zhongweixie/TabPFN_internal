#!/usr/bin/env python
"""Phase 1b-ish: GLOBAL cross-dataset indexer with training-set LOO labels.

Addresses the leak-free Phase-1a failure (held-out AUC 0.466), where the indexer
saw only 60 val queries from one dataset. Hypotheses tested here:
  (1) more + more-diverse supervision (cross-dataset) lets the indexer learn a
      GENERALIZABLE row-selection rule, vs single-dataset sample starvation;
  (2) the selection rule is shared across datasets (TabPFN-style), so ONE global
      indexer can serve all.

NO LEAKAGE: per dataset we split into context / query / test (all disjoint, none
from the real test set used for golden labels). golden labels come from
training-set LOO on the QUERY rows (query rows are NOT in their own context, so
they play the test-row role exactly, but with KNOWN labels — legitimate since they
are not the held-out test rows). Eval is on the held-out test rows only.

Train ONE indexer on pooled (query_repr_d, context_repr_d, golden_d) across datasets;
evaluate per-dataset on held-out test. GO if held-out indexer > full AND > KNN.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
import run_indexer_gate as G  # noqa: E402
from run_loo_ceiling import loo_influence  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("indexer_global_results.log", mode="w")],
)
log = logging.getLogger("indexer_global")

DEVICE = "cuda"
N_CTX = 120          # context (train) rows per dataset
N_QUERY = 80         # query rows for training-set LOO golden labels
N_TEST = 60          # held-out eval rows
KEEP_FRAC = G.KEEP_FRAC
SEED = 0

# __APPEND_MARKER__


def make_samples(name, X, y, seed):
    """Split into context/query/test (disjoint). golden labels via training-set LOO on
    QUERY rows (fit on context, query rows not in context). Returns dict of tensors."""
    labels = np.unique(y)
    Xc, Xrest, yc, yrest = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    Xq, Xte, yq, yte = train_test_split(Xrest, yrest, test_size=0.5, random_state=seed, stratify=yrest)
    Xc, yc = Xc[:N_CTX], yc[:N_CTX]
    Xq, yq = Xq[:N_QUERY], yq[:N_QUERY]
    Xte, yte = Xte[:N_TEST], yte[:N_TEST]

    clf = G.mk_clf(seed); clf.fit(Xc, yc)
    # training-set LOO: influence of each context row j on each query row q (no test leak)
    infl, _ = loo_influence(Xc, yc, Xq, yq, seed)        # (Nq, Nctx)
    Nc = len(Xc); keep = max(1, int(round(Nc * KEEP_FRAC)))
    golden = np.zeros((len(Xq), Nc), dtype=np.float32)
    for m in range(len(Xq)):
        golden[m, np.argsort(-infl[m])[:keep]] = 1.0

    # reprs: context-row keys + query-row queries (for training)
    dq = G.extract_layer_reprs(clf, Xq)
    ctx_repr = dq[:Nc].float(); query_repr = dq[Nc:].float()
    # reprs for held-out test eval (context keys recomputed alongside)
    dte = G.extract_layer_reprs(clf, Xte)
    ctx_repr_te = dte[:Nc].float(); test_repr = dte[Nc:].float()
    return dict(name=name, clf=clf, labels=labels,
                Xc=Xc, yc=yc, Xte=Xte, yte=yte,
                ctx_repr=ctx_repr, query_repr=query_repr, golden=golden,
                ctx_repr_te=ctx_repr_te, test_repr=test_repr)


def train_global_indexer(samples, epochs=400, lr=1e-3):
    """One indexer trained on pooled (query_repr, ctx_repr, golden) across datasets.

    Each dataset has its OWN context, so scores are computed per-dataset and the loss
    summed — the indexer params are shared (the generalizable selection rule)."""
    dev = samples[0]["query_repr"].device
    idx = G.RowIndexer().to(dev)
    opt = torch.optim.AdamW(idx.parameters(), lr=lr, weight_decay=1e-4)
    labs = [torch.tensor(s["golden"], device=dev) for s in samples]
    idx.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for s, lab in zip(samples, labs):
            logits = idx(s["query_repr"], s["ctx_repr"])   # (Nq, Nctx) for THIS dataset
            loss = loss + G.focal_bce(logits, lab)
        loss = loss / len(samples)
        loss.backward(); opt.step()
    idx.eval()
    return idx


def evaluate(idx, samples):
    log.info("=" * 64)
    for s in samples:
        clf, labels = s["clf"], s["labels"]
        Xte, yte = s["Xte"], s["yte"]
        full = G._score(yte, clf.predict_proba(Xte), labels)[1]
        sc = idx.scores(s["test_repr"], s["ctx_repr_te"])     # held-out test queries
        ix = G._score(yte, G.predict_with_indexer(clf, Xte, sc), labels)[1]
        # context-as-train KNN baseline + oracle ceiling on test (uses yte, ref only)
        knn = G._score(yte, G.knn_topk_proba(clf, s["Xc"], s["yc"], Xte, KEEP_FRAC, labels), labels)[1]
        from run_loo_ceiling import eval_per_query_topk
        infl_te, _ = loo_influence(s["Xc"], s["yc"], Xte, yte, SEED)
        ceil = G._score(yte, eval_per_query_topk(s["Xc"], s["yc"], Xte, yte, infl_te, KEEP_FRAC, SEED, labels), labels)[1]
        log.info("%-14s full=%.3f ceil(oracle)=%.3f indexer(heldout)=%.3f knn=%.3f | "
                 "beats_full=%s beats_knn=%s",
                 s["name"], full, ceil, ix, knn, ix > full, ix > knn)


def main():
    log.info("GLOBAL cross-dataset indexer | ctx=%d query=%d test=%d keep=%.0f%% | "
             "training-set LOO labels, held-out test", N_CTX, N_QUERY, N_TEST, KEEP_FRAC * 100)
    raw = [("breast_cancer", *load_breast_cancer(return_X_y=True))]
    for nm in ["phoneme", "electricity", "qsar-biodeg"]:
        try:
            d = fetch_openml(nm, version=1, as_frame=False, parser="liac-arff")
            X = d.data.astype("float32"); cls = np.unique(d.target)
            raw.append((nm, X, np.searchsorted(cls, d.target).astype(int)))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", nm, str(e)[:60])
    t0 = time.time()
    samples = [make_samples(nm, X, y, SEED) for nm, X, y in raw]
    log.info("built %d dataset samples in %.1fs", len(samples), time.time() - t0)
    idx = train_global_indexer(samples)
    evaluate(idx, samples)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()


