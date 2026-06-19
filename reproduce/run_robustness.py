#!/usr/bin/env python
"""Robustness checks answering the reviewers' two main critiques of the NO-GO gates:

  (A) ADAPTIVE DEPTH was only tested clean / moderate-N / K=32 / 50-50 cross-fit.
      The earlier LOO ceiling gate found denoising headroom ONLY under label noise.
      Here: inject train label-flip noise, raise K_MAX to 64, AND sweep the cross-fit
      decide/eval split ratio (50/50, 75/25) to rule out "halving K is too harsh".
      If a snoop-free oracle STILL can't beat uniform under noise+big-K -> NO-GO holds.

  (B) CALIBRATION head was only SCALAR temperature (argmax-preserving, weakest knob).
      Here: VECTOR (per-class) temperature head T_{i,c}=Tg_c*exp(delta_{i,c}), far more
      capacity to express per-row structure, plus relaxed reg (wd=0, wider clamp, more
      epochs). If per-row vector scaling STILL can't beat a per-CLASS global vector on
      held-out NLL/ECE -> calibration NO-GO holds beyond the scalar case.

Reuses the audited functions from run_adaptive_ceiling.py and run_calib_head.py.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import run_adaptive_ceiling as AC
import run_calib_head as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("robustness_results.log", mode="w")],
)
log = logging.getLogger("robustness")

SEEDS = [0, 1, 2]
EPS = 1e-7

# __APPEND_MARKER__


def flip(y, labels, frac, rng):
    y = y.copy(); m = int(len(y) * frac)
    idx = rng.choice(len(y), m, replace=False)
    for i in idx:
        y[i] = rng.choice([c for c in labels if c != y[i]])
    return y


# ---------- (A) adaptive depth under noise + big-K + split sweep ----------

def xfit_ratio(p, col, B, decide_frac, rng):
    """Cross-fit oracle with arbitrary decide/eval split. Decide budget on `decide_frac`
    of the K members, score on the rest. Returns (acc_adapt, ll_adapt, acc_unif, ll_unif)
    on the EVAL members at matched average budget."""
    K, M, Cn = p.shape
    perm = rng.permutation(K)
    nd = max(2, int(round(K * decide_frac)))
    d_idx, e_idx = perm[:nd], perm[nd:]
    cm_d = AC.running_means(p[d_idx]); cm_e = AC.running_means(p[e_idx])
    Ke = cm_e.shape[0]
    loss_d = AC.row_loss(cm_d, col)
    b = AC.alloc_oracle(loss_d, min(B * M, cm_d.shape[0] * M))
    b = np.clip(b, 1, Ke)
    pred = cm_e[b - 1, np.arange(M), :]
    from sklearn.metrics import accuracy_score, log_loss
    aa = accuracy_score(col, pred.argmax(1))
    la = log_loss(col, pred, labels=list(range(Cn)))
    bu = np.clip(AC.alloc_uniform(M, int(round(b.mean()))), 1, Ke)
    pu = cm_e[bu - 1, np.arange(M), :]
    au = accuracy_score(col, pu.argmax(1))
    lu = log_loss(col, pu, labels=list(range(Cn)))
    return aa, la, au, lu


def run_adaptive_noise(name, X, y, seed, flip_frac, k_max, n_train=1500, n_test=300):
    rng = np.random.RandomState(seed)
    labels = np.unique(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    if len(Xtr) > n_train:
        i = rng.choice(len(Xtr), n_train, replace=False); Xtr, ytr = Xtr[i], ytr[i]
    if len(Xte) > n_test:
        i = rng.choice(len(Xte), n_test, replace=False); Xte, yte = Xte[i], yte[i]
    if flip_frac > 0:
        ytr = flip(ytr, labels, flip_frac, rng)
    clf = AC.TabPFNClassifier(n_estimators=k_max, device=AC.DEVICE, model_path=AC.MODEL_PATH,
                              random_state=seed, fit_mode="fit_preprocessors",
                              ignore_pretraining_limits=True)
    clf.fit(Xtr, ytr)
    p = AC.member_probs(clf, Xte)
    ci = {c: i for i, c in enumerate(clf.classes_)}
    col = np.array([ci[c] for c in yte])
    out = {}
    for B in [2, 4, 8, 16]:
        r50 = xfit_ratio(p, col, B, 0.5, np.random.RandomState(seed + 11))
        r75 = xfit_ratio(p, col, B, 0.75, np.random.RandomState(seed + 13))
        out[B] = {"xf50": r50, "xf75": r75}
    return out


def section_A():
    log.info("#" * 80)
    log.info("(A) ADAPTIVE DEPTH under label noise + K_MAX=64 + cross-fit split sweep")
    datasets = AC.load_datasets()
    for flip_frac in [0.0, 0.1, 0.25]:
        log.info("=" * 80)
        log.info(">>> train label-flip = %.0f%%", flip_frac * 100)
        for name, X, y in datasets:
            t0 = time.time()
            runs = [run_adaptive_noise(name, X, y, s, flip_frac, 64) for s in SEEDS]
            for B in [2, 4, 8, 16]:
                a50 = np.mean([r[B]["xf50"][0] - r[B]["xf50"][2] for r in runs])
                l50 = np.mean([r[B]["xf50"][1] - r[B]["xf50"][3] for r in runs])
                a75 = np.mean([r[B]["xf75"][0] - r[B]["xf75"][2] for r in runs])
                l75 = np.mean([r[B]["xf75"][1] - r[B]["xf75"][3] for r in runs])
                log.info("  %-13s B=%2d | xfit50 Δacc=%+.3f Δll=%+.3f | "
                         "xfit75 Δacc=%+.3f Δll=%+.3f", name, B, a50, l50, a75, l75)
            log.info("  (%s, %.1fs)", name, time.time() - t0)
    log.info("READ(A): if snoop-free Δacc stays ~0/negative even at 25%% noise & 75/25 "
             "split => adaptive-depth NO-GO is robust, not a clean/small-K artifact.")


# ---------- (B) vector (per-class) temperature head ----------

def fit_global_vecT(logits, col, n_classes):
    """Per-class global temperature vector Tg_c = argmin val-NLL, fit per class via grid.
    Stronger baseline than scalar T: each class gets its own scalar."""
    grid = np.concatenate([np.linspace(0.3, 1.0, 15), np.linspace(1.0, 6.0, 26)])
    Tg = np.ones(n_classes)
    # coordinate descent: a couple of sweeps over classes
    for _ in range(3):
        for c in range(n_classes):
            best, bestnll = Tg[c], 1e9
            for t in grid:
                Tc = Tg.copy(); Tc[c] = t
                z = logits / Tc[None, :]
                z = z - z.max(1, keepdims=True)
                lp = z - np.log(np.exp(z).sum(1, keepdims=True))
                v = -lp[np.arange(len(col)), col].mean()
                if v < bestnll:
                    bestnll, best = v, t
            Tg[c] = best
    return Tg


def vec_nll_ece(logits, col, Tvec):
    """Tvec: (C,) global per-class OR (M,C) per-row per-class."""
    T = Tvec[None, :] if Tvec.ndim == 1 else Tvec
    z = logits / T; z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    nll = float(-np.log(np.clip(p[np.arange(len(col)), col], EPS, 1)).mean())
    conf = p.max(1); pred = p.argmax(1); acc = (pred == col).astype(float)
    edges = np.linspace(0, 1, 16); e = 0.0
    for i in range(15):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return nll, float(e)


class VecTempHead(torch.nn.Module):
    """Per-row, per-class temperature: T_{i,c} = Tg_c * exp(delta_{i,c}). Zero-init -> at
    start equals the per-class global vector (strong baseline). wd=0, wide clamp."""

    def __init__(self, n_feat, n_classes, logTg):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feat, 64), torch.nn.GELU(),
            torch.nn.Linear(64, n_classes))
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
        self.register_buffer("logTg", torch.tensor(logTg, dtype=torch.float32))

    def forward(self, f):
        return torch.exp(self.logTg[None, :] + self.net(f).clamp(-3.0, 3.0))  # (M,C)


def run_calib_vec(name, X, y, seed):
    rng = np.random.RandomState(seed)
    labels = np.unique(y)
    Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xr, yr, test_size=0.5, random_state=seed, stratify=yr)
    Xtr, ytr = Xtr[:1000], ytr[:1000]; Xva, yva = Xva[:300], yva[:300]; Xte, yte = Xte[:300], yte[:300]
    clf = AC.mk(seed); clf.fit(Xtr, ytr)
    ci = {c: i for i, c in enumerate(clf.classes_)}; ncl = len(clf.classes_)
    ml_va, p_va = C.mean_logits(clf, Xva); col_va = np.array([ci[c] for c in yva])
    ml_te, p_te = C.mean_logits(clf, Xte); col_te = np.array([ci[c] for c in yte])
    F_va, F_te = C.row_features(ml_va, p_va), C.row_features(ml_te, p_te)

    Tg = fit_global_vecT(ml_va, col_va, ncl)                 # per-class global vector
    # train vector head on val NLL
    f = torch.tensor((F_va - F_va.mean(0)) / (F_va.std(0) + 1e-6), dtype=torch.float32)
    mu, sd = F_va.mean(0), F_va.std(0) + 1e-6
    lg = torch.tensor(ml_va, dtype=torch.float32); yv = torch.tensor(col_va)
    head = VecTempHead(F_va.shape[1], ncl, np.log(Tg))
    opt = torch.optim.Adam(head.net.parameters(), lr=5e-3)   # wd=0
    head.train()
    for _ in range(800):
        opt.zero_grad()
        T = head(f).clamp(1e-2, 50.0)
        z = lg / T
        loss = torch.nn.functional.cross_entropy(z, yv)
        loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        T_te = head(torch.tensor((F_te - mu) / sd, dtype=torch.float32)).clamp(1e-2, 50.0).numpy()
    g = vec_nll_ece(ml_te, col_te, Tg)
    v = vec_nll_ece(ml_te, col_te, T_te)
    return {"nll_g": g[0], "nll_v": v[0], "ece_g": g[1], "ece_v": v[1],
            "Tspread": float(T_te.std())}


def section_B():
    log.info("#" * 80)
    log.info("(B) VECTOR (per-class) temperature head vs per-class GLOBAL vector")
    for name, X, y in AC.load_datasets():
        t0 = time.time()
        runs = [run_calib_vec(name, X, y, s) for s in SEEDS]
        ng = np.mean([r["nll_g"] for r in runs]); nv = np.mean([r["nll_v"] for r in runs])
        eg = np.mean([r["ece_g"] for r in runs]); ev = np.mean([r["ece_v"] for r in runs])
        ts = np.mean([r["Tspread"] for r in runs])
        log.info("  %-13s NLL g=%.4f v=%.4f (Δ=%+.4f) | ECE g=%.4f v=%.4f (Δ=%+.4f) | "
                 "per-row T std=%.3f (%.1fs)", name, ng, nv, nv - ng, eg, ev, ev - eg,
                 ts, time.time() - t0)
    log.info("READ(B): Δ<0 on NLL&ECE => vector per-row calibration helps (GO). ~0 with "
             "T std~0 => even per-class per-row temp collapses to the global vector "
             "(NO-GO holds beyond scalar).")


def main():
    section_A()
    section_B()
    log.info("ALL DONE")


if __name__ == "__main__":
    main()
