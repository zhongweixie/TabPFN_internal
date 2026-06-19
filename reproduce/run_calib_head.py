#!/usr/bin/env python
"""Plan C — gate: DECOUPLED PER-ROW CALIBRATION HEAD.

The last untested FlashMemory-flavoured angle: a tiny decoupled head on the FROZEN
TabPFN that predicts a PER-ROW softmax temperature T_i (and/or a per-row blend), to
improve calibration (NLL / ECE) without touching the backbone.

Why this is structurally different from the two falsified directions:
  - row-selection / adaptive-depth tried to predict per-row structure (which rows,
    how much compute) that TabPFN has already collapsed -> no stable signal.
  - calibration is POST-HOC on the final logits. Temperature scaling T>0 PRESERVES
    argmax, so ACCURACY IS MATHEMATICALLY UNCHANGED. This is purely an NLL/ECE play.
    The only question is whether per-row T beats a single global T.

THE HONEST GATE (snoop-free):
  Baselines fit on a VAL split, evaluated on a DISJOINT TEST split:
    (0) uncalibrated (T=1)
    (1) GLOBAL temperature  T* = argmin val-NLL   (the standard, strong baseline)
  Candidate:
    (2) CONDITIONAL temperature  T_i = head(features_i), head trained on val, eval test.
        features = per-row uncertainty signals from the FROZEN model only
        (predictive entropy, max-prob/margin, ensemble-member disagreement, layer-12
        rep norm) — NO labels at inference, NO test leak.
  GO  if conditional-T consistently beats global-T on held-out TEST NLL & ECE.
  NO-GO if conditional-T ~= global-T  (a single scalar already captures all the
        calibration error there is -> no per-row calibration structure to learn).

Same one-fit trick as adaptive-depth: member_probs gives per-member probs (K,M,C) so we
get the ensemble-mean logits AND member-disagreement features from a single fit.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split

import run_adaptive_ceiling as AC   # reuse mk(), member_probs(), running_means()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("calib_head_results.log", mode="w")],
)
log = logging.getLogger("calib_head")

DEVICE = "cuda"
SEEDS = [0, 1]
N_TRAIN = 1000
N_VAL = 300          # held-out for fitting global-T and the conditional head
N_TEST = 300         # disjoint eval (never seen by either calibrator)
K_MAX = 32
EPS = 1e-7
N_BINS = 15          # ECE bins

# __APPEND_MARKER__


def mean_logits(clf, X):
    """Ensemble-mean logits (M,C) + per-member probs (K,M,C) from ONE fit.

    We calibrate the ensemble-MEAN logit (the thing predict_proba softmaxes), and use
    the member probs only to build disagreement FEATURES (frozen, label-free)."""
    raw = clf.predict_raw_logits(X)                 # (K,M,C) class-aligned
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 4:
        raw = raw[:, :, 0, :]
    ml = raw.mean(0)                                # (M,C) mean logit over members
    e = np.exp(raw - raw.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)                # (K,M,C) per-member probs
    return ml, p


def nll(logits, col, T):
    """Mean NLL of temperature-scaled logits. logits:(M,C), col:(M,), T scalar or (M,)."""
    z = logits / (T[:, None] if np.ndim(T) else T)
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(col)), col].mean())


def ece(logits, col, T, n_bins=N_BINS):
    """Expected Calibration Error of temperature-scaled probs."""
    z = logits / (T[:, None] if np.ndim(T) else T)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    conf = p.max(1); pred = p.argmax(1); acc = (pred == col).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def fit_global_T(logits, col):
    """T* = argmin NLL via 1-D grid + local refine (convex in 1/T, grid is plenty)."""
    grid = np.concatenate([np.linspace(0.3, 1.0, 15), np.linspace(1.0, 6.0, 26)])
    nlls = [nll(logits, col, float(t)) for t in grid]
    return float(grid[int(np.argmin(nlls))])


def row_features(ml, p):
    """Per-row frozen, label-free uncertainty features (M,F). No backbone touch.

    The calibration head sees ONLY these — exactly what is available at deployment."""
    e = np.exp(ml - ml.max(1, keepdims=True)); pm = e / e.sum(1, keepdims=True)  # (M,C)
    sort = np.sort(pm, 1)
    maxp = sort[:, -1]
    margin = sort[:, -1] - sort[:, -2]
    ent = -(pm * np.log(np.clip(pm, EPS, 1))).sum(1)        # predictive entropy
    member_ent = -(p * np.log(np.clip(p, EPS, 1))).sum(-1)  # (K,M)
    disagree_std = member_ent.std(0)                        # member entropy spread
    member_maxp = p.max(-1).std(0)                          # member max-prob spread
    logit_gap = np.sort(ml, 1)[:, -1] - np.sort(ml, 1)[:, -2]
    F = np.stack([maxp, margin, ent, disagree_std, member_maxp, logit_gap], 1)
    return F.astype(np.float64)


class TempHead(torch.nn.Module):
    """Tiny MLP: row features -> per-row temperature T_i > 0.

    T_i = T_global * exp(small delta), so the head only has to learn the per-row
    DEVIATION from the global optimum — at init T_i == T_global (identity to the strong
    baseline), so it can only help if a per-row signal exists."""

    def __init__(self, n_feat, logT_global):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feat, 32), torch.nn.GELU(),
            torch.nn.Linear(32, 1),
        )
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
        self.logT0 = torch.nn.Parameter(torch.tensor(float(logT_global)), requires_grad=False)

    def forward(self, feat):
        delta = self.net(feat).squeeze(-1).clamp(-2.0, 2.0)   # bounded deviation
        return torch.exp(self.logT0 + delta)                  # (M,) positive temperature


def fit_cond_head(F_val, logits_val, col_val, T_global, epochs=300, lr=3e-3):
    """Train the temperature head on VAL to minimise VAL NLL. Frozen features in."""
    dev = "cpu"  # tiny problem; cpu avoids host<->gpu churn
    f = torch.tensor(F_val, dtype=torch.float32, device=dev)
    mu, sd = f.mean(0, keepdim=True), f.std(0, keepdim=True) + 1e-6
    f = (f - mu) / sd
    lg = torch.tensor(logits_val, dtype=torch.float32, device=dev)
    y = torch.tensor(col_val, dtype=torch.long, device=dev)
    head = TempHead(F_val.shape[1], np.log(T_global)).to(dev)
    opt = torch.optim.Adam(head.net.parameters(), lr=lr, weight_decay=1e-3)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        T = head(f).clamp(1e-2, 50.0)
        z = lg / T[:, None]
        loss = torch.nn.functional.cross_entropy(z, y)
        loss.backward(); opt.step()
    head.eval()
    return head, (mu.numpy(), sd.numpy())


def cond_T(head, norm, F):
    mu, sd = norm
    f = torch.tensor((F - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        return head(f).clamp(1e-2, 50.0).numpy()


def run_one(name, X, y, seed):
    rng = np.random.RandomState(seed)
    labels = np.unique(y)
    Xtr, Xrest, ytr, yrest = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(
        Xrest, yrest, test_size=0.5, random_state=seed, stratify=yrest)
    if len(Xtr) > N_TRAIN:
        i = rng.choice(len(Xtr), N_TRAIN, replace=False); Xtr, ytr = Xtr[i], ytr[i]
    if len(Xva) > N_VAL:
        i = rng.choice(len(Xva), N_VAL, replace=False); Xva, yva = Xva[i], yva[i]
    if len(Xte) > N_TEST:
        i = rng.choice(len(Xte), N_TEST, replace=False); Xte, yte = Xte[i], yte[i]

    clf = AC.mk(seed); clf.fit(Xtr, ytr)
    ci = {c: i for i, c in enumerate(clf.classes_)}
    ml_va, p_va = mean_logits(clf, Xva); col_va = np.array([ci[c] for c in yva])
    ml_te, p_te = mean_logits(clf, Xte); col_te = np.array([ci[c] for c in yte])
    F_va, F_te = row_features(ml_va, p_va), row_features(ml_te, p_te)

    Tg = fit_global_T(ml_va, col_va)                    # global-T fit on VAL
    head, norm = fit_cond_head(F_va, ml_va, col_va, Tg)
    T_te = cond_T(head, norm, F_te)                     # per-row T on held-out TEST

    return {
        "T_global": Tg, "T_cond_mean": float(T_te.mean()), "T_cond_std": float(T_te.std()),
        "nll": (nll(ml_te, col_te, 1.0), nll(ml_te, col_te, Tg), nll(ml_te, col_te, T_te)),
        "ece": (ece(ml_te, col_te, 1.0), ece(ml_te, col_te, Tg), ece(ml_te, col_te, T_te)),
    }


def aggregate(name, runs):
    log.info("=" * 78)
    log.info("DATASET %s  (n_seeds=%d, K_MAX=%d)", name, len(runs), K_MAX)
    nll0 = np.mean([r["nll"][0] for r in runs]); nllg = np.mean([r["nll"][1] for r in runs])
    nllc = np.mean([r["nll"][2] for r in runs])
    ec0 = np.mean([r["ece"][0] for r in runs]); ecg = np.mean([r["ece"][1] for r in runs])
    ecc = np.mean([r["ece"][2] for r in runs])
    Tg = np.mean([r["T_global"] for r in runs]); Tcm = np.mean([r["T_cond_mean"] for r in runs])
    Tcs = np.mean([r["T_cond_std"] for r in runs])
    log.info("  NLL  uncal=%.4f  global-T=%.4f  cond-T=%.4f  | cond−global=%+.4f",
             nll0, nllg, nllc, nllc - nllg)
    log.info("  ECE  uncal=%.4f  global-T=%.4f  cond-T=%.4f  | cond−global=%+.4f",
             ec0, ecg, ecc, ecc - ecg)
    log.info("  T_global=%.2f  T_cond mean=%.2f std=%.2f (spread of per-row T)",
             Tg, Tcm, Tcs)
    log.info("  READ: cond−global < 0 on BOTH NLL & ECE on held-out test => per-row "
             "calibration headroom (GO). ~0 => a single scalar already captures it (NO-GO).")


def main():
    log.info("CALIBRATION-HEAD gate | N_tr=%d N_val=%d N_te=%d K_MAX=%d seeds=%s",
             N_TRAIN, N_VAL, N_TEST, K_MAX, SEEDS)
    for name, X, y in AC.load_datasets():
        t0 = time.time()
        runs = [run_one(name, X, y, s) for s in SEEDS]
        aggregate(name, runs)
        log.info("  (%.1fs)", time.time() - t0)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()
