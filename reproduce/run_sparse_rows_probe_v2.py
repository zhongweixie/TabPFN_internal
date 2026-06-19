#!/usr/bin/env python
"""Phase 0 probe v2 (rebuilt after adversarial review): sparse training-row selection.

Fixes over v1 (all four review findings):
  (1) FAITHFUL multi-query attention. v1 reimplemented the test-row path using only
      head-0 and broadcast (head-0 top-k overlapped other heads only ~12% -> "attn"
      results were invalid). v2 keeps ALL H query heads attending to the selected
      single-kv-head subset, and selects rows by attention MASS aggregated over all
      heads. keep=N reproduces the original forward byte-for-byte.
  (2) ORACLE ceiling. A 2-pass mode: pass 1 accumulates test->train attention mass
      across ALL layers+heads into a global importance matrix; pass 2 restricts every
      layer to the global top-k per test row. This is the ceiling for attention-defined
      importance (does sparse selection have ANY headroom?).
  (3) NOISE-INJECTION denoising test. Optionally inject irrelevant training rows
      (random features + random labels); a real denoiser should filter them and
      recover accuracy. Clean-data results alone don't test the denoising hypothesis.
  (4) PROPER metrics + multi-seed. accuracy + balanced accuracy + ROC-AUC, 3 seeds,
      mean±std. (Raw accuracy alone misleads on imbalanced sets like phoneme.)

Selection modes (per test row, choose which train rows it attends to):
  full   : attend all train rows (= unmodified model)
  attn   : per-layer, top-k by THIS layer's multi-head-aggregated attention mass (online/H2O-like)
  oracle : global top-k by all-layer accumulated attention mass (ceiling)
  knn    : top-k by L2 in the shared input-embedding space (LoCalPFN-style baseline)
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
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
        logging.FileHandler("sparse_rows_probe_v2_results.log", mode="w"),
    ],
)
log = logging.getLogger("sparse_probe_v2")

DEVICE = "cuda"
MODEL_PATH = (
    "/home/zxiebk/workspace/model/tabpfn_2_5/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)
SEEDS = [0, 1, 2]
KEEP_FRACS = [0.10, 0.25, 0.50]
TRAIN_CAP = 8000          # applied equally to all methods; documented, not silent

# Runtime control + oracle accumulator (set per evaluation).
_CFG = {"mode": "full", "keep_frac": 1.0, "phase": None}
_ORACLE = {"imp": None}   # accumulates (Bc, M, N) test->train attention mass
_orig_forward = v25.AlongColumnAttention.forward

# __APPEND_MARKER__


def _test_attn_weights_BcMN(q_test_BcMHD, k0_BcND, head_dim):
    """Multi-head attention weights of each test row over train rows, aggregated
    (mean) over query heads. q_test: (Bc,M,H,D) all query heads; k0: (Bc,N,D) the
    single kv head test rows use. Returns mass (Bc,M,N) = mean_h softmax_n(q_h·k0)."""
    # scores per head: (Bc, M, H, N)
    scores = torch.einsum("bmhd,bnd->bmhn", q_test_BcMHD.float(), k0_BcND.float())
    scores = scores * (head_dim ** -0.5)
    w = torch.softmax(scores, dim=-1)          # softmax over train rows N
    return w.mean(dim=2)                        # mean over heads -> (Bc, M, N)


def _multihead_attend_subset(q_test_BcMHD, k0_BcND, v0_BcND, idx_BcMk, head_dim):
    """Faithful multi-query attention: all H query heads attend to the selected
    subset of the single kv head. idx_BcMk: (Bc,M,keep) train indices per test row.
    Returns (Bc, M, H, D)."""
    Bc, M, H, D = q_test_BcMHD.shape
    keep = idx_BcMk.shape[-1]
    gi = idx_BcMk.unsqueeze(-1).expand(-1, -1, -1, D)              # (Bc,M,keep,D)
    k_sel = torch.gather(k0_BcND.unsqueeze(1).expand(-1, M, -1, -1), 2, gi)
    v_sel = torch.gather(v0_BcND.unsqueeze(1).expand(-1, M, -1, -1), 2, gi)
    # scores: (Bc,M,H,keep); softmax over keep; weighted sum of v_sel (shared kv head)
    scores = torch.einsum("bmhd,bmkd->bmhk", q_test_BcMHD, k_sel) * (head_dim ** -0.5)
    w = torch.softmax(scores, dim=-1)
    out = torch.einsum("bmhk,bmkd->bmhd", w, v_sel)               # (Bc,M,H,D)
    return out


def patched_forward(self, x_BcRE, single_eval_pos=None, *, cached_kv=None,
                    return_kv=False):
    mode, keep_frac, phase = _CFG["mode"], _CFG["keep_frac"], _CFG["phase"]
    Bc, R, _ = x_BcRE.shape
    active = (
        cached_kv is None and not return_kv
        and single_eval_pos is not None and 0 < single_eval_pos < R
        and (mode in ("attn", "knn") or (mode == "oracle" and phase in ("collect", "apply")))
    )
    if not active:
        return _orig_forward(self, x_BcRE, single_eval_pos,
                             cached_kv=cached_kv, return_kv=return_kv)

    N = single_eval_pos
    M = R - N
    q_BcRHD = self.q_projection(x_BcRE).view(Bc, R, -1, self.head_dim)
    k_BcNHD = self.k_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
    v_BcNHD = self.v_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
    H = q_BcRHD.shape[2]

    # Train rows: unchanged full multi-head attention (identical to original).
    out_train = scaled_dot_product_attention(q_BcRHD[:, :N], k_BcNHD, v_BcNHD)

    q_test = q_BcRHD[:, N:]              # (Bc, M, H, D)
    k0 = k_BcNHD[:, :, 0]               # (Bc, N, D) single kv head
    v0 = v_BcNHD[:, :, 0]

    # --- Oracle: pass 1 accumulates importance, runs FULL attention (no pruning) ---
    if mode == "oracle" and phase == "collect":
        mass = _test_attn_weights_BcMN(q_test, k0, self.head_dim)   # (Bc,M,N)
        cur = _ORACLE["imp"]
        _ORACLE["imp"] = mass if cur is None else cur + mass
        return _orig_forward(self, x_BcRE, single_eval_pos,
                             cached_kv=cached_kv, return_kv=return_kv)

    keep = max(1, int(round(N * keep_frac)))

    # --- choose indices per test row ---
    if mode == "attn":
        mass = _test_attn_weights_BcMN(q_test, k0, self.head_dim)
        idx = mass.topk(keep, dim=-1).indices
    elif mode == "knn":
        emb_t = x_BcRE[:, N:].float(); emb_n = x_BcRE[:, :N].float()
        idx = (-torch.cdist(emb_t, emb_n)).topk(keep, dim=-1).indices
    elif mode == "oracle" and phase == "apply":
        imp = _ORACLE["imp"]                     # (Bc,M,N) global, all-layer
        idx = imp.topk(min(keep, imp.shape[-1]), dim=-1).indices
    else:
        raise ValueError((mode, phase))

    out_test = _multihead_attend_subset(q_test, k0, v0, idx, self.head_dim)
    output_BcRHD = torch.cat([out_train, out_test], dim=1)
    return self.out_projection(output_BcRHD.reshape(Bc, R, -1)), None


def install():
    v25.AlongColumnAttention.forward = patched_forward


def uninstall():
    v25.AlongColumnAttention.forward = _orig_forward


def metrics(y_true, proba, labels):
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_true, pred)
    bacc = balanced_accuracy_score(y_true, pred)
    try:
        if len(labels) == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                labels=labels)
    except ValueError:
        auc = float("nan")
    return acc, bacc, auc


def predict_mode(clf, X_te, mode, keep_frac):
    """Run predict_proba under a given selection mode. Oracle uses 2 passes."""
    if mode == "oracle":
        _ORACLE["imp"] = None
        _CFG.update(mode="oracle", keep_frac=keep_frac, phase="collect")
        clf.predict_proba(X_te)
        _CFG.update(mode="oracle", keep_frac=keep_frac, phase="apply")
        proba = clf.predict_proba(X_te)
        _ORACLE["imp"] = None
        return proba
    _CFG.update(mode=mode, keep_frac=keep_frac, phase=None)
    return clf.predict_proba(X_te)


def inject_noise(X_tr, y_tr, labels, frac, rng):
    """Append `frac`*n irrelevant rows: random features (from per-feature marginals)
    + random labels. Returns augmented (X, y)."""
    n = X_tr.shape[0]
    n_noise = int(round(n * frac))
    if n_noise == 0:
        return X_tr, y_tr
    # sample each feature independently from its empirical marginal (VIP-COP S1_noi)
    cols = [rng.choice(X_tr[:, j], size=n_noise) for j in range(X_tr.shape[1])]
    X_noise = np.stack(cols, axis=1).astype(X_tr.dtype)
    y_noise = rng.choice(labels, size=n_noise)
    return np.vstack([X_tr, X_noise]), np.concatenate([y_tr, y_noise])


def build_clf(seed):
    return TabPFNClassifier(
        n_estimators=1, device=DEVICE, model_path=MODEL_PATH, random_state=seed,
        fit_mode="fit_preprocessors", ignore_pretraining_limits=True,
    )


def run_one(name, X, y, seed, noise_frac):
    labels = np.unique(y)
    rng = np.random.RandomState(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    if X_tr.shape[0] > TRAIN_CAP:
        idx = rng.choice(X_tr.shape[0], TRAIN_CAP, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    if noise_frac > 0:
        X_tr, y_tr = inject_noise(X_tr, y_tr, labels, noise_frac, rng)

    clf = build_clf(seed)
    clf.fit(X_tr, y_tr)
    res = {}
    res[("full", 1.0)] = metrics(y_te, predict_mode(clf, X_te, "full", 1.0), labels)
    for mode in ("attn", "knn", "oracle"):
        for kf in KEEP_FRACS:
            res[(mode, kf)] = metrics(y_te, predict_mode(clf, X_te, mode, kf), labels)
    return res


def aggregate(name, runs, noise_frac):
    log.info("##### %s  noise=%.0f%%  (n_seeds=%d) #####",
             name, noise_frac * 100, len(runs))
    keys = list(runs[0].keys())
    for k in keys:
        accs = np.array([r[k][0] for r in runs])
        baccs = np.array([r[k][1] for r in runs])
        aucs = np.array([r[k][2] for r in runs])
        mode, kf = k
        log.info("  %-6s k=%3.0f%%  acc=%.3f±%.3f  bacc=%.3f±%.3f  auc=%.3f±%.3f",
                 mode, kf * 100, accs.mean(), accs.std(),
                 baccs.mean(), baccs.std(), np.nanmean(aucs), np.nanstd(aucs))


def load_datasets():
    out = []
    for nm in ["phoneme", "electricity", "jungle_chess_2pcs_raw_endgame_complete"]:
        try:
            d = fetch_openml(nm, version=1, as_frame=False, parser="liac-arff")
            X = d.data.astype("float32")
            cls = np.unique(d.target)
            y = np.searchsorted(cls, d.target).astype(int)
            out.append((nm, X, y))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", nm, str(e)[:80])
    return out


def main():
    log.info("Sparse-row probe v2 | modes=full/attn/knn/oracle | keep=%s | "
             "seeds=%s | cap=%d | metrics=acc/bacc/auc", KEEP_FRACS, SEEDS, TRAIN_CAP)
    install()
    try:
        for name, X, y in load_datasets():
            log.info("=" * 72)
            log.info("DATASET %s  shape=%s  classes=%d", name, X.shape, len(np.unique(y)))
            for noise_frac in (0.0, 0.5):     # clean, then 50% injected noise rows
                t0 = time.time()
                runs = [run_one(name, X, y, s, noise_frac) for s in SEEDS]
                aggregate(name, runs, noise_frac)
                log.info("  (%.1fs)", time.time() - t0)
    finally:
        uninstall()
    log.info("ALL DONE")


if __name__ == "__main__":
    main()



