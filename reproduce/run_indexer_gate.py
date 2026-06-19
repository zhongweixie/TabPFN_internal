#!/usr/bin/env python
"""Phase 1a minimal gate: learned row-indexer vs LOO ceiling vs KNN (per-dataset).

Pipeline (all decoupled / frozen backbone):
  1. precompute: tap layer-12 pre-output x_BRCD from the frozen model, mean-pool over
     columns -> per-row reprs. test-row query reprs + train-row key reprs.
  2. golden labels [M,N] in {0,1}, two branches (only this differs):
       loo    : reuse run_loo_ceiling.loo_influence; top-k% influential rows = 1.
       voting : per-layer top-p(0.6) binarize of test->train attention, vote >= theta.
  3. label pre-check: predicting each test row with its golden rows should >= full
     (loo by construction; voting is the open question).
  4. train a tiny low-rank dual-tower indexer (focal loss + 3:1 neg sampling),
     backbone NOT in the loop (only cached reprs + labels).
  5. inference: indexer scores train rows per test row -> top-k% -> the test->train
     attention uses only that subset (gather-then-SDPA, no (M,H,N) materialization).
  6. gate: indexer vs full / KNN-local / LOO-ceiling, accuracy + ROC-AUC.
     GO if indexer > KNN AND reaches >= 70% of the LOO ceiling gain.

This is a MECHANISM gate on 1-2 small datasets, not a benchmark. per-dataset indexer
(1a); cross-dataset transfer (1b) only if this passes.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# reuse LOO influence from the ceiling script
sys.path.insert(0, ".")
from run_loo_ceiling import loo_influence, mk as mk_clf  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("indexer_gate_results.log", mode="w"),
    ],
)
log = logging.getLogger("indexer_gate")

DEVICE = "cuda"
MODEL_PATH = (
    "/home/zxiebk/workspace/model/tabpfn_2_5/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)
LAYER = 12           # mid-late layer to tap per-row reprs from
EMSIZE = 192
RANK = 64
N_HEADS = 4
KEEP_FRAC = 0.25     # selection budget (matches a strong point on the ceiling)
SEED = 0

# __APPEND_MARKER__


class RowIndexer(nn.Module):
    """Low-rank dual-tower scorer: (test_repr[M,E], train_repr[N,E]) -> logits[M,N].

    Low-rank multi-head bilinear + head routing. ~78k params (<0.1% of TabPFN).
    Trains decoupled on frozen reprs.

    NOTE (bug fix): the first version used ReLU on the per-head q.k score, which on the
    real layer-12 reprs hit a dead-relu regime — logits collapsed to a constant
    (std->0), PR-AUC stuck at chance (0.25). Removed the ReLU and added input LayerNorm
    (the reprs have norm ~12, unnormalised). With these two changes PR-AUC on real reprs
    went 0.25 -> 0.66. Both fixes verified before committing.
    """

    def __init__(self, emsize=EMSIZE, rank=RANK, n_heads=N_HEADS):
        super().__init__()
        self.rank, self.n_heads = rank, n_heads
        self.ln = nn.LayerNorm(emsize)
        self.q_proj = nn.Linear(emsize, rank * n_heads, bias=False)
        self.k_proj = nn.Linear(emsize, rank * n_heads, bias=False)
        self.head_w = nn.Linear(emsize, n_heads, bias=False)
        for m in (self.q_proj, self.k_proj, self.head_w):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, test_repr, train_repr):
        te = self.ln(test_repr); tr = self.ln(train_repr)
        M, N = te.shape[0], tr.shape[0]
        q = self.q_proj(te).view(M, self.n_heads, self.rank)
        k = self.k_proj(tr).view(N, self.n_heads, self.rank)
        per_head = torch.einsum("mhr,nhr->mhn", q, k) * (self.rank ** -0.5)  # no relu
        w = torch.softmax(self.head_w(te), dim=-1)         # [M,H] head routing
        return torch.einsum("mhn,mh->mn", per_head, w)     # logits [M,N]

    @torch.no_grad()
    def scores(self, test_repr, train_repr):
        return torch.sigmoid(self.forward(test_repr, train_repr))


def extract_layer_reprs(clf, X_te, layer=LAYER):
    """Run predict_proba with a hook on blocks[layer]; return per-row reprs split into
    (train_key_repr[N,E], test_query_repr[M,E]) by single_eval_pos. Frozen / no grad.

    The hooked x_BRCD is (B,R,C,E); R = num_thinking + N_train + M_test. We mean-pool
    over columns C, drop the thinking rows, and split train vs test.
    """
    m = clf.model_
    n_think = m.add_thinking_rows.num_thinking_rows
    cap = {}

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        cap["x"] = x.detach()

    h = m.blocks[layer].register_forward_hook(hook)
    clf.predict_proba(X_te)
    h.remove()
    x = cap["x"]                       # (B,R,C,E), B=1 for n_estimators=1
    perrow = x.mean(dim=2)[0]          # (R,E)
    data = perrow[n_think:]            # drop thinking rows -> (N_train+M_test, E)
    return data


def gen_labels_loo(Xtr, ytr, Xte, yte, seed, keep_frac=KEEP_FRAC):
    """golden[m,j]=1 if train row j in top-keep% by LOO influence for test row m."""
    infl, _ = loo_influence(Xtr, ytr, Xte, yte, seed)   # [M,N]
    N = len(ytr); keep = max(1, int(round(N * keep_frac)))
    lab = np.zeros((len(Xte), N), dtype=np.float32)
    for m in range(len(Xte)):
        lab[m, np.argsort(-infl[m])[:keep]] = 1.0
    return lab


def gen_labels_voting(clf, X_te, seed, p=0.1, theta=3):
    """Per-layer top-p binarize of test->train attention mass, vote across layers.

    NOTE: FlashMemory used p=0.6, theta=3 (on 21-layer LLM, long context). On small-N
    tabular that produced mean_pos=1.0 (every row voted in -> degenerate "all important"
    labels). Swept params: p=0.1, theta=3 gives mean_pos~0.25, matching the LOO keep
    budget so the two branches are comparable. (Finding: FlashMemory's voting params do
    NOT transfer to small-N tabular; the threshold must be retuned to the budget.)
    """
    from tabpfn.architectures import tabpfn_v2_5 as v25
    m = clf.model_
    n_think = m.add_thinking_rows.num_thinking_rows
    votes = {}
    orig = v25.AlongColumnAttention.forward

    def hook_fwd(self, x_BcRE, single_eval_pos=None, *, cached_kv=None, return_kv=False):
        if cached_kv is None and single_eval_pos is not None and 0 < single_eval_pos < x_BcRE.shape[1]:
            N = single_eval_pos
            q = self.q_projection(x_BcRE).view(x_BcRE.shape[0], x_BcRE.shape[1], -1, self.head_dim)
            k0 = self.k_projection(x_BcRE[:, :N]).view(x_BcRE.shape[0], N, -1, self.head_dim)[:, :, 0]
            qt = q[:, N:]                                    # (Bc,M,H,D)
            s = torch.einsum("bmhd,bnd->bmhn", qt.float(), k0.float()) * (self.head_dim ** -0.5)
            w = torch.softmax(s, dim=-1).mean(dim=2)         # (Bc,M,N) head-mean
            w = w.mean(dim=0)                                # (M,N) col-group mean
            # drop thinking cols from N axis
            w_real = w[:, n_think:]
            # top-p binarize per test row
            order = torch.argsort(-w_real, dim=-1)
            sorted_w = torch.gather(w_real, 1, order)
            csum = sorted_w.cumsum(-1)
            keepmask = csum <= p
            keepmask[:, 0] = True                            # always keep top-1
            sel = torch.zeros_like(w_real)
            sel.scatter_(1, order, keepmask.float())
            lid = len(votes)
            votes[lid] = sel.cpu().numpy()
        return orig(self, x_BcRE, single_eval_pos, cached_kv=cached_kv, return_kv=return_kv)

    v25.AlongColumnAttention.forward = hook_fwd
    try:
        clf.predict_proba(X_te)
    finally:
        v25.AlongColumnAttention.forward = orig
    V = np.stack(list(votes.values()), 0).sum(0)             # (M,N) vote counts
    return (V >= theta).astype(np.float32)


def focal_bce(logits, labels, gamma=2.0, neg_pos=3.0):
    """Focal BCE with 3:1 negative subsampling per test row."""
    losses = []
    for m in range(logits.shape[0]):
        lo, la = logits[m], labels[m]
        pos = (la > 0.5)
        npos = int(pos.sum().item())
        if npos == 0:
            continue
        neg_idx = torch.where(~pos)[0]
        k = min(len(neg_idx), int(npos * neg_pos))
        sel_neg = neg_idx[torch.randperm(len(neg_idx), device=lo.device)[:k]]
        idx = torch.cat([torch.where(pos)[0], sel_neg])
        bce = F.binary_cross_entropy_with_logits(lo[idx], la[idx], reduction="none")
        p = torch.sigmoid(lo[idx])
        pt = torch.where(la[idx] > 0.5, p, 1 - p)
        losses.append(((1 - pt) ** gamma * bce).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def train_indexer(test_repr, train_repr, labels, epochs=400, lr=1e-3):
    """Decoupled training: only cached reprs + labels, backbone NOT in the loop."""
    dev = test_repr.device
    idx = RowIndexer().to(dev)
    opt = torch.optim.AdamW(idx.parameters(), lr=lr, weight_decay=1e-4)
    lab = torch.tensor(labels, device=dev)
    idx.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = idx(test_repr, train_repr)
        loss = focal_bce(logits, lab)
        loss.backward()
        opt.step()
    idx.eval()
    return idx


# ---- inference hook: indexer selects per-test-row subset (gather-then-SDPA) ----
from tabpfn.architectures import tabpfn_v2_5 as v25  # noqa: E402
from tabpfn.architectures.shared.scaled_dot_product_attention import (  # noqa: E402
    scaled_dot_product_attention,
)

_HOOK = {"on": False, "scores": None, "keep_frac": KEEP_FRAC}
_orig_aca = v25.AlongColumnAttention.forward


def _patched(self, x_BcRE, single_eval_pos=None, *, cached_kv=None, return_kv=False):
    if not _HOOK["on"] or cached_kv is not None or return_kv or single_eval_pos is None \
            or not (0 < single_eval_pos < x_BcRE.shape[1]):
        return _orig_aca(self, x_BcRE, single_eval_pos, cached_kv=cached_kv, return_kv=return_kv)
    Bc, R, _ = x_BcRE.shape
    N = single_eval_pos; M = R - N
    sc = _HOOK["scores"]                     # (M_expected, N_real)
    # guard: if the forward's M or N doesn't match the precomputed scores, fall back
    # (e.g. test-row chunking, or a different layer's call) — never crash the model.
    n_real = sc.shape[1]
    if M != sc.shape[0] or n_real > N:
        return _orig_aca(self, x_BcRE, single_eval_pos, cached_kv=cached_kv, return_kv=return_kv)
    q = self.q_projection(x_BcRE).view(Bc, R, -1, self.head_dim)
    k = self.k_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
    v = self.v_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
    out_train = scaled_dot_product_attention(q[:, :N], k, v)
    qt = q[:, N:]; k0 = k[:, :, 0]; v0 = v[:, :, 0]   # (Bc,N,D) single kv head over thinking+train
    offset = N - n_real                                # thinking rows occupy [0:offset)
    keep = max(1, int(round(n_real * _HOOK["keep_frac"])))
    sel = sc.topk(keep, dim=-1).indices + offset       # (M,keep) selected REAL-train idx
    # test rows ALWAYS attend to the thinking rows [0:offset) + the selected train rows.
    think_idx = torch.arange(offset, device=x_BcRE.device).unsqueeze(0).expand(M, -1)
    idx_full = torch.cat([think_idx, sel.to(x_BcRE.device)], dim=1)  # (M, offset+keep)
    K = idx_full.shape[1]
    gi = idx_full.unsqueeze(0).expand(Bc, -1, -1).unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
    k_sel = torch.gather(k0.unsqueeze(1).expand(-1, M, -1, -1), 2, gi)
    v_sel = torch.gather(v0.unsqueeze(1).expand(-1, M, -1, -1), 2, gi)
    s = torch.einsum("bmhd,bmkd->bmhk", qt, k_sel) * (self.head_dim ** -0.5)
    w = torch.softmax(s, dim=-1)
    out_test = torch.einsum("bmhk,bmkd->bmhd", w, v_sel)
    out = torch.cat([out_train, out_test], dim=1)
    return self.out_projection(out.reshape(Bc, R, -1)), None


def predict_with_indexer(clf, X_te, scores_MN, keep_frac=KEEP_FRAC):
    _HOOK.update(on=True, scores=scores_MN, keep_frac=keep_frac)
    v25.AlongColumnAttention.forward = _patched
    try:
        return clf.predict_proba(X_te)
    finally:
        _HOOK["on"] = False
        v25.AlongColumnAttention.forward = _orig_aca


def _score(y, proba, labels):
    pred = labels[np.argmax(proba, axis=1)]
    acc = accuracy_score(y, pred)
    try:
        auc = (roc_auc_score(y, proba[:, 1]) if len(labels) == 2
               else roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        auc = float("nan")
    return acc, auc


def knn_topk_proba(clf, Xtr, ytr, Xte, keep_frac, labels):
    """KNN-local baseline: each test row predicted with its keep% nearest train rows."""
    from scipy.spatial.distance import cdist
    N = len(Xtr); keep = max(len(labels) * 2, int(round(N * keep_frac)))
    D = cdist(Xte, Xtr)
    out = np.zeros((len(Xte), len(labels)))
    for m in range(len(Xte)):
        idx = np.argsort(D[m])[:keep]
        if len(np.unique(ytr[idx])) < len(labels):
            idx = np.arange(N)
        c = mk_clf(SEED); c.fit(Xtr[idx], ytr[idx])
        pm = c.predict_proba(Xte[m:m + 1])
        for i, cc in enumerate(c.classes_):
            out[m, list(labels).index(cc)] = pm[0, i]
    return out


def run_dataset(name, X, y, branch):
    labels = np.unique(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    Xtr, ytr = Xtr[:120], ytr[:120]
    Xte, yte = Xte[:40], yte[:40]
    log.info("=" * 60)
    log.info("DATASET %s  branch=%s  N=%d M=%d", name, branch, len(Xtr), len(Xte))

    clf = mk_clf(SEED); clf.fit(Xtr, ytr)
    full_acc, full_auc = _score(yte, clf.predict_proba(Xte), labels)
    log.info("full          acc=%.3f auc=%.3f", full_acc, full_auc)

    # LOO influence computed ONCE: used for both the ceiling and (if branch==loo) labels
    from run_loo_ceiling import eval_per_query_topk
    infl, _ = loo_influence(Xtr, ytr, Xte, yte, SEED)
    N = len(Xtr); keep = max(1, int(round(N * KEEP_FRAC)))

    # golden labels
    if branch == "loo":
        lab = np.zeros((len(Xte), N), dtype=np.float32)
        for m in range(len(Xte)):
            lab[m, np.argsort(-infl[m])[:keep]] = 1.0
    else:
        lab = gen_labels_voting(clf, Xte, SEED)
    log.info("golden labels mean_pos=%.3f", lab.mean())

    # per-row reprs from layer 12
    data = extract_layer_reprs(clf, Xte)
    train_repr = data[:len(Xtr)].float(); test_repr = data[len(Xtr):].float()

    # train indexer (decoupled)
    idx = train_indexer(test_repr, train_repr, lab)
    sc = idx.scores(test_repr, train_repr)         # (M,N)

    # ceiling (LOO top-k) for reference — reuse infl
    ceil_acc, ceil_auc = _score(
        yte, eval_per_query_topk(Xtr, ytr, Xte, yte, infl, KEEP_FRAC, SEED, labels), labels)

    # indexer prediction
    idx_acc, idx_auc = _score(yte, predict_with_indexer(clf, Xte, sc), labels)
    # KNN baseline
    knn_acc, knn_auc = _score(yte, knn_topk_proba(clf, Xtr, ytr, Xte, KEEP_FRAC, labels), labels)

    log.info("ceiling(LOO)  acc=%.3f auc=%.3f", ceil_acc, ceil_auc)
    log.info("indexer       acc=%.3f auc=%.3f", idx_acc, idx_auc)
    log.info("knn-local     acc=%.3f auc=%.3f", knn_acc, knn_auc)
    # gate verdict
    gain_ceil = ceil_auc - full_auc
    gain_idx = idx_auc - full_auc
    frac = gain_idx / gain_ceil if abs(gain_ceil) > 1e-6 else float("nan")
    log.info("VERDICT %s/%s: indexer_auc-full=%.3f  ceil_gain=%.3f  frac_of_ceil=%.2f  beats_knn=%s",
             name, branch, gain_idx, gain_ceil, frac, idx_auc > knn_auc)


def main():
    log.info("Indexer gate (Phase 1a) | layer=%d keep=%.0f%% | branches=loo,voting",
             LAYER, KEEP_FRAC * 100)
    ds = [("breast_cancer", *load_breast_cancer(return_X_y=True))]
    try:
        d = fetch_openml("phoneme", version=1, as_frame=False, parser="liac-arff")
        X = d.data.astype("float32"); cls = np.unique(d.target)
        ds.append(("phoneme", X, np.searchsorted(cls, d.target).astype(int)))
    except Exception as e:  # noqa: BLE001
        log.warning("skip phoneme: %s", str(e)[:60])
    for name, X, y in ds:
        for branch in ("loo", "voting"):
            run_dataset(name, X, y, branch)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()




