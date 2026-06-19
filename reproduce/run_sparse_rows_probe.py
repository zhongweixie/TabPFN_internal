#!/usr/bin/env python
"""Phase 0 probe for the sparse-row-selection direction: "less is more" test.

Question (go/no-go): on a large training set, if each TEST row attends to only its
top-k% most-relevant TRAINING rows (instead of all of them), does accuracy
  (a) hold / improve  -> "less is more" denoising holds -> worth a learned indexer
  (b) collapse        -> no-go.
And: is per-query ATTENTION-based selection better than per-query KNN selection?
(KNN = LoCalPFN-style local context — the real baseline, per VIP-COP findings.)

This is TRAINING-FREE and NON-INVASIVE: we monkey-patch AlongColumnAttention.forward
in v2.5 to, on the test-row path, score q_test · k_train^T and keep only the top-k
train rows per test row (mask the rest), instead of attending to all train rows.

Selection modes:
  - "attn": top-k train rows by attention score q_test·k_train  (white-box, ours)
  - "knn":  top-k train rows by feature-space L2 distance       (LoCalPFN-style baseline)
  - "full": baseline, attend all train rows (k=100%)            (= unmodified model)

We sweep keep-fraction k and compare attn vs knn vs full on accuracy + NLL.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier
from tabpfn.architectures import tabpfn_v2_5 as v25
from tabpfn.architectures.shared.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sparse_rows_probe_results.log", mode="w"),
    ],
)
log = logging.getLogger("sparse_probe")

DEVICE = "cuda"
MODEL_PATH = (
    "/home/zxiebk/workspace/model/tabpfn_2_5/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)
SEED = 0
KEEP_FRACS = [0.05, 0.10, 0.25, 0.50, 1.00]

# __APPEND_MARKER__

# Runtime controls read by the patched attention (set per evaluation).
_CFG = {"keep_frac": 1.0, "mode": "full"}
_orig_forward = v25.AlongColumnAttention.forward


def _topk_indices(scores_BcMN: torch.Tensor, keep: int) -> torch.Tensor:
    """Top-`keep` train-row indices per (Bc, test-row), by score. Returns (Bc, M, keep)."""
    return scores_BcMN.topk(keep, dim=-1).indices


def patched_forward(self, x_BcRE, single_eval_pos=None, *, cached_kv=None,
                    return_kv=False):
    """AlongColumnAttention.forward with optional per-test-row top-k train selection.

    Falls back to the original implementation whenever the sparse path doesn't apply
    (cached_kv path, return_kv, no eval split, or keep_frac>=1 / mode='full').
    """
    keep_frac = _CFG["keep_frac"]
    mode = _CFG["mode"]
    Bc, R, _ = x_BcRE.shape
    sparse_applies = (
        mode != "full"
        and keep_frac < 1.0
        and cached_kv is None
        and not return_kv
        and single_eval_pos is not None
        and 0 < single_eval_pos < R
    )
    if not sparse_applies:
        return _orig_forward(self, x_BcRE, single_eval_pos,
                             cached_kv=cached_kv, return_kv=return_kv)

    N = single_eval_pos                # train (+thinking) rows
    M = R - N                          # test rows
    keep = max(1, int(round(N * keep_frac)))

    q_BcRHD = self.q_projection(x_BcRE).view(Bc, R, -1, self.head_dim)
    k_BcNHD = self.k_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
    v_BcNHD = self.v_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)

    # Train rows: unchanged full attention among themselves.
    out_train = scaled_dot_product_attention(q_BcRHD[:, :N], k_BcNHD, v_BcNHD)

    # Test rows: multi-query (first kv head), pick top-k train rows per test row.
    q_test_BcM1D = q_BcRHD[:, N:, :1]                      # (Bc, M, 1, D)
    k1_BcN1D = k_BcNHD[:, :, :1]                           # (Bc, N, 1, D)
    v1_BcN1D = v_BcNHD[:, :, :1]
    qd = q_test_BcM1D.squeeze(2).squeeze(1) if False else q_test_BcM1D[:, :, 0]  # (Bc,M,D)
    kd = k1_BcN1D[:, :, 0]                                 # (Bc, N, D)

    if mode == "attn":
        scores_BcMN = torch.einsum("bmd,bnd->bmn", qd, kd) * (self.head_dim ** -0.5)
    elif mode == "knn":
        # Fair KNN: L2 in the SHARED input-embedding space x_BcRE (same projection
        # for train & test), NOT across the different q/k projections. Higher score =
        # closer neighbor.
        emb_test = x_BcRE[:, N:].float()                  # (Bc, M, E)
        emb_train = x_BcRE[:, :N].float()                 # (Bc, N, E)
        scores_BcMN = -torch.cdist(emb_test, emb_train)   # (Bc, M, N)
    else:
        raise ValueError(mode)

    idx_BcMk = _topk_indices(scores_BcMN, keep)            # (Bc, M, keep)
    # Gather the selected train K/V per test row -> (Bc, M, keep, D)
    gather_idx = idx_BcMk.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
    k_sel = torch.gather(
        kd.unsqueeze(1).expand(-1, M, -1, -1), 2, gather_idx)   # (Bc,M,keep,D)
    v_sel = torch.gather(
        v1_BcN1D[:, :, 0].unsqueeze(1).expand(-1, M, -1, -1), 2, gather_idx)

    # Per-test-row attention over its own selected subset. Flatten (Bc*M) as batch.
    q_flat = qd.reshape(Bc * M, 1, 1, self.head_dim)
    k_flat = k_sel.reshape(Bc * M, keep, 1, self.head_dim)
    v_flat = v_sel.reshape(Bc * M, keep, 1, self.head_dim)
    out_test = scaled_dot_product_attention(q_flat, k_flat, v_flat)
    out_test_BcM1D = out_test.reshape(Bc, M, 1, self.head_dim)
    # broadcast single-head output back to all query heads (multi-query)
    H = q_BcRHD.shape[2]
    out_test_BcMHD = out_test_BcM1D.expand(-1, -1, H, -1)

    output_BcRHD = torch.cat([out_train, out_test_BcMHD], dim=1)
    output_BcRE = self.out_projection(
        output_BcRHD.reshape(Bc, R, -1)
    )
    return output_BcRE, None


def install_patch() -> None:
    v25.AlongColumnAttention.forward = patched_forward


def remove_patch() -> None:
    v25.AlongColumnAttention.forward = _orig_forward


def build_clf(seed: int) -> TabPFNClassifier:
    return TabPFNClassifier(
        n_estimators=1, device=DEVICE, model_path=MODEL_PATH,
        random_state=seed, fit_mode="fit_preprocessors",
        ignore_pretraining_limits=True,
    )


def evaluate(clf, X_te, y_te, labels, mode, keep_frac) -> tuple[float, float, float]:
    _CFG["mode"], _CFG["keep_frac"] = mode, keep_frac
    t0 = time.time()
    proba = clf.predict_proba(X_te)
    dt = time.time() - t0
    acc = accuracy_score(y_te, np.argmax(proba, axis=1))
    pc = np.clip(proba, 1e-7, 1.0)
    pc = pc / pc.sum(axis=1, keepdims=True)
    ll = log_loss(y_te, pc, labels=labels)
    return acc, ll, dt


def run_dataset(name: str, X, y) -> None:
    labels = np.unique(y)
    log.info("=" * 72)
    log.info("DATASET %s  shape=%s  classes=%d", name, X.shape, len(labels))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    clf = build_clf(SEED)
    clf.fit(X_tr, y_tr)

    # baseline: full attention (unmodified)
    acc0, ll0, dt0 = evaluate(clf, X_te, y_te, labels, "full", 1.0)
    log.info("  [full ]            acc=%.4f  nll=%.4f  t=%.2fs  (n_train=%d)",
             acc0, ll0, dt0, len(X_tr))

    for mode in ("attn", "knn"):
        for kf in KEEP_FRACS:
            if kf >= 1.0:
                continue
            acc, ll, dt = evaluate(clf, X_te, y_te, labels, mode, kf)
            log.info("  [%-4s k=%4.0f%%] acc=%.4f (%+.4f)  nll=%.4f  t=%.2fs",
                     mode, kf * 100, acc, acc - acc0, ll, dt)


def load_datasets():
    out = []
    # moderately large so the N^2 train attention + selection is meaningful
    for nm in ["phoneme", "electricity", "jungle_chess_2pcs_raw_endgame_complete"]:
        try:
            d = fetch_openml(nm, version=1, as_frame=False, parser="liac-arff")
            X = d.data.astype("float32")
            cls = np.unique(d.target)
            y = (np.searchsorted(cls, d.target)).astype(int)
            # cap to keep the probe fast but still "large train"
            if X.shape[0] > 12000:
                idx = np.random.RandomState(SEED).choice(X.shape[0], 12000, replace=False)
                X, y = X[idx], y[idx]
            out.append((nm, X, y))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", nm, str(e)[:80])
    return out


def main() -> None:
    log.info("Sparse-row-selection probe | keep_fracs=%s | per-query attn vs knn",
             KEEP_FRACS)
    install_patch()
    try:
        for name, X, y in load_datasets():
            run_dataset(name, X, y)
    finally:
        remove_patch()
    log.info("ALL DONE")


if __name__ == "__main__":
    main()


