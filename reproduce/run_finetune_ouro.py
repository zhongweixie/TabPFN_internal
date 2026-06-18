#!/usr/bin/env python
"""Branch B (thinking/ouro): Ouro-style looped fine-tune of TabPFN v2.5.

Faithful to Ouro (Scaling Latent Reasoning via Looped Language Models):
  - Recurrence = FULL-stack weight-shared loop (not latent-only).
  - LEARNABLE per-step residual gate (_ouro_step_gate): step 0 is the full
    pretrained forward; extra steps are gated residual refinements x <- x +
    tanh(g_k)*(stack(x)-x). Gate init 0 => T>1 reproduces T=1 exactly, so the
    recurrence starts as a near-identity map (cures Phase 0's full-stack collapse).
  - PER-STEP output supervision: decode the test rows at EVERY step, weight each
    step's cross-entropy by an exit distribution, plus an entropy regulariser so
    the model doesn't trivially dump all mass on the last step.
  - Fixed T_max = 4 during training (Ouro's setting).

Contrast with Branch A (coconut): A carries state only on latent rows with
final-pass-only supervision; here the whole hidden state recurs (gated) and every
step is supervised. Phase 0 showed naked full-stack recurrence collapses; the gate
+ per-step loss are exactly what Ouro uses to make it monotone instead.
"""

from __future__ import annotations

import logging
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_wine,
)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tabpfn import TabPFNClassifier
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.finetuning.data_util import (
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("finetune_ouro_results.log", mode="w"),
    ],
)
log = logging.getLogger("ouro_ft")

DEVICE = "cuda"
MODEL_PATH = (
    "/home/zxiebk/workspace/model/tabpfn_2_5/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)
SEEDS = [0, 1, 2]
EVAL_STEPS = [1, 2, 4, 8]
T_MAX = 4                  # fixed training recurrence depth (Ouro)
EPOCHS = 12
LR = 1e-5
GATE_LR = 1e-3            # the few gate scalars can use a faster LR
QUERY_SPLIT = 0.3
TARGET_CHUNKS = 6
ENTROPY_BETA = 0.05       # entropy reg on the exit distribution (Ouro uses 0.05-0.1)


def chunk_size_for(n_train: int) -> int:
    size = max(40, n_train // TARGET_CHUNKS)
    return min(size, n_train - 1)


def build_estimator(seed: int) -> TabPFNClassifier:
    return TabPFNClassifier(
        n_estimators=1,
        device=DEVICE,
        model_path=MODEL_PATH,
        random_state=seed,
        fit_mode="fit_preprocessors",
    )


def set_thinking(clf: TabPFNClassifier, steps: int) -> None:
    clf.model_._thinking_steps = steps
    clf.model_._thinking_mode = "ouro"


def ensure_gate(clf: TabPFNClassifier) -> torch.nn.Parameter:
    """Lazily attach the learnable per-step gate (T_MAX-1 extra steps)."""
    m = clf.model_
    g = getattr(m, "_ouro_step_gate", None)
    if g is None:
        emb = m.add_thinking_rows.row_token_values_TE
        g = torch.nn.Parameter(torch.zeros(T_MAX - 1, device=emb.device, dtype=emb.dtype))
        m._ouro_step_gate = g
    return g


def ouro_loss(step_logits: list[torch.Tensor], targets_BQ: torch.Tensor):
    """Per-step weighted cross-entropy + entropy reg over a UNIFORM-prior exit
    distribution. step_logits[k] is decode output at step k, shape (Q, B, 1, L)-ish
    from _decode -> (M, B, L). We reduce each to a per-step CE, then weight by a
    softmax over negative losses (cheap learned-free exit proxy) regularised toward
    uniform via entropy.
    """
    per_step_ce = []
    for lg in step_logits:
        # _decode returns test_output_MB1 shape (M, B, L); reshape to (B*?, L, M)
        M, B, L = lg.shape
        logits_BLM = lg.permute(1, 2, 0).reshape(B, L, M)
        ce = F.cross_entropy(logits_BLM, targets_BQ[:B])
        per_step_ce.append(ce)
    ce_stack = torch.stack(per_step_ce)  # (T,)
    # exit distribution: prefer lower-loss steps, but regularise toward uniform
    w = torch.softmax(-ce_stack.detach(), dim=0)
    weighted = (w * ce_stack).sum()
    entropy = -(w * (w + 1e-9).log()).sum()
    return weighted - ENTROPY_BETA * entropy, ce_stack.detach()


def evaluate(clf: TabPFNClassifier, X_te, y_te, labels, tag: str) -> dict:
    out = {}
    for steps in EVAL_STEPS:
        set_thinking(clf, steps)
        proba = clf.predict_proba(X_te)
        acc = accuracy_score(y_te, np.argmax(proba, axis=1))
        pc = np.clip(proba, 1e-7, 1.0)
        pc = pc / pc.sum(axis=1, keepdims=True)
        ll = log_loss(y_te, pc, labels=labels)
        drift = getattr(clf.model_, "_thinking_drift", None)
        dstr = (" drift=[" + ", ".join(f"{d[1]:.1f}" for d in drift) + "]") if drift else ""
        log.info("  [%s] steps=%d  acc=%.4f  nll=%.4f%s", tag, steps, acc, ll, dstr)
        out[steps] = (acc, ll)
    set_thinking(clf, 1)
    return out


def finetune_one(name: str, X, y, seed: int) -> dict:
    labels = np.unique(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    base = build_estimator(seed)
    base.fit(X_tr, y_tr)
    log.info("--- [%s seed=%d] BEFORE fine-tune ---", name, seed)
    pre = evaluate(base, X_te, y_te, labels, "pre")

    est = build_estimator(seed)
    est.fit(X_tr, y_tr)
    model = est.model_
    model.train()
    set_thinking(est, T_MAX)
    gate = ensure_gate(est)
    model._thinking_collect_step_logits = True
    perf = PerformanceOptions(
        save_peak_memory_factor=None,
        force_recompute_layer=False,
        use_chunkwise_inference=False,
        enable_torch_compile=False,
    )
    # gate gets its own faster param group; the rest of the model the base LR.
    other = [p for n, p in model.named_parameters() if "_ouro_step_gate" not in n]
    optim = torch.optim.AdamW(
        [{"params": other, "lr": LR}, {"params": [gate], "lr": GATE_LR}],
        weight_decay=0.01,
    )
    max_chunk = chunk_size_for(len(X_tr))

    for epoch in range(EPOCHS):
        epoch_seed = 100 * seed + epoch
        splitter = partial(
            train_test_split, test_size=QUERY_SPLIT, random_state=epoch_seed
        )
        ds = get_preprocessed_dataset_chunks(
            calling_instance=est,
            X_raw=X_tr,
            y_raw=y_tr,
            split_fn=splitter,
            max_data_size=max_chunk,
            model_type="classifier",
            equal_split_size=False,
            data_shuffle_seed=epoch_seed,
            preprocessing_random_state=epoch_seed,
        )
        if len(ds) == 0:
            continue
        loader = DataLoader(
            ds, batch_size=1, collate_fn=meta_dataset_collator, shuffle=True
        )
        ep_loss, nb, last_ce = 0.0, 0, None
        for batch in loader:
            ctx_u = torch.unique(
                torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_context])
            )
            qry_u = torch.unique(batch.y_query.reshape(-1))
            if not bool(torch.isin(qry_u, ctx_u, assume_unique=True).all()):
                continue
            optim.zero_grad()
            est.fit_from_preprocessed(
                batch.X_context,
                batch.y_context,
                batch.cat_indices,
                batch.configs,
                performance_options=perf,
            )
            _ = est.forward(batch.X_query, return_raw_logits=True)
            step_logits = model._thinking_step_logits
            targets = batch.y_query.reshape(1, -1).to(DEVICE).long()
            loss, ce_stack = ouro_loss(step_logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += float(loss.detach().item())
            last_ce = ce_stack
            nb += 1
        ce_str = (
            " per_step_ce=[" + ", ".join(f"{c:.3f}" for c in last_ce.tolist()) + "]"
            if last_ce is not None
            else ""
        )
        gate_str = "[" + ", ".join(f"{torch.tanh(g).item():+.2f}" for g in gate) + "]"
        log.info(
            "  epoch %2d  batches=%d  loss=%.4f  gate(tanh)=%s%s",
            epoch,
            nb,
            ep_loss / max(nb, 1),
            gate_str,
            ce_str,
        )

    # snapshot trained weights (incl. gate), rebuild predict executor, restore
    model.eval()
    model._thinking_collect_step_logits = False
    trained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gate_snapshot = gate.detach().clone()
    est.fit(X_tr, y_tr)
    est.model_.load_state_dict(trained_state, strict=False)
    est.model_._ouro_step_gate = torch.nn.Parameter(gate_snapshot)
    log.info("--- [%s seed=%d] AFTER fine-tune ---", name, seed)
    post = evaluate(est, X_te, y_te, labels, "post")
    return {"pre": pre, "post": post}


def load_datasets():
    out = []
    for nm, ld in [
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ]:
        d = ld()
        out.append((nm, d.data, d.target))
    for nm in ["phoneme", "qsar-biodeg"]:
        try:
            d = fetch_openml(nm, version=1, as_frame=False, parser="liac-arff")
            X = d.data.astype("float32")
            classes = np.unique(d.target)
            y = (d.target == classes[1]).astype(int) if len(classes) == 2 else d.target
            out.append((nm, X, y))
        except Exception as e:  # noqa: BLE001
            log.warning("skip OpenML %s: %s", nm, str(e)[:80])
    return out


def aggregate_and_log(name: str, runs: list[dict]) -> None:
    log.info("##### SUMMARY %s (n_seeds=%d) #####", name, len(runs))
    base = np.mean([r["pre"][1][0] for r in runs])
    for phase in ("pre", "post"):
        for steps in EVAL_STEPS:
            accs = np.array([r[phase][steps][0] for r in runs])
            nlls = np.array([r[phase][steps][1] for r in runs])
            log.info(
                "  %-4s steps=%d  acc=%.4f±%.4f  nll=%.4f±%.4f",
                phase, steps, accs.mean(), accs.std(), nlls.mean(), nlls.std(),
            )
    for steps in EVAL_STEPS:
        d = np.array([r["post"][steps][0] - r["pre"][steps][0] for r in runs])
        log.info(
            "  Δacc(post-pre) steps=%d: %+.4f±%.4f  (base pre@1=%.4f)",
            steps, d.mean(), d.std(), base,
        )


def main() -> None:
    log.info(
        "Ouro looped fine-tune | T_max=%d | gated residual | per-step CE + "
        "entropy(beta=%.2f) | seeds=%s",
        T_MAX, ENTROPY_BETA, SEEDS,
    )
    for name, X, y in load_datasets():
        log.info("=" * 72)
        log.info("DATASET %s  shape=%s  classes=%d", name, X.shape, len(np.unique(y)))
        runs = []
        t0 = time.time()
        for seed in SEEDS:
            runs.append(finetune_one(name, X, y, seed))
        aggregate_and_log(name, runs)
        log.info("%s done in %.1fs", name, time.time() - t0)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()

