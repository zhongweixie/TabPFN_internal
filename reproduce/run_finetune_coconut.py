#!/usr/bin/env python
"""Branch A (thinking/coconut): COCONUT-style looped fine-tune of TabPFN v2.5.

Faithful to COCONUT (Chain of Continuous Thought), NO gating:
  - Recurrence = latent-only feedback. v2.5's learnable "thinking rows" play the
    role of COCONUT's continuous-thought tokens; the train+test data rows are the
    fixed "question" context, reset to their initial embedding each pass.
  - CURRICULUM (the soul of COCONUT): we map "progressively replace text reasoning
    steps with latent tokens" -> "progressively increase the number of latent loop
    passes". stage 0 = 1 pass, then 2, then TARGET_STEPS.
  - Supervision: cross-entropy on the final-pass query predictions only.
  - Trainable: full model incl. add_thinking_rows.row_token_values_TE.

We hand-roll a compact training loop (reusing TabPFN's data pipeline
get_preprocessed_dataset_chunks + fit_from_preprocessed + forward) so the curriculum
schedule and multi-depth evaluation are fully under our control. The batched
training engine feeds cat([X_ctx, X_qry]) as one full table to model.forward (plain,
non-cache path), so the thinking loop fires during training and gradients flow
through every pass.

Goal: does looped fine-tune (a) keep the drift from exploding, taming it toward
convergence, and (b) make deeper inference (more steps) help rather than hurt?
"""

from __future__ import annotations

import logging
import sys
import time
from functools import partial

import numpy as np
import torch
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
from tabpfn.finetuning.finetuned_classifier import _compute_classification_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("finetune_coconut_results.log", mode="w"),
    ],
)
log = logging.getLogger("coconut_ft")

DEVICE = "cuda"
MODEL_PATH = (
    "/home/zxiebk/workspace/model/tabpfn_2_5/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)
SEEDS = [0, 1, 2]                  # multi-seed for statistical validity
EVAL_STEPS = [1, 2, 4, 8]          # inference depths to probe before/after FT
TARGET_STEPS = 4                   # max curriculum depth
EPOCHS = 12
EPOCHS_PER_STAGE = 4               # stage 0:steps=1, 1:steps=2, 2:steps=4
LR = 1e-5
QUERY_SPLIT = 0.3                  # fraction of each chunk used as query
# We want MULTIPLE chunks/epoch (several gradient steps), so we size each chunk as
# a fraction of the training set rather than a fixed count. get_preprocessed_dataset_
# chunks returns 0 chunks if max_data_size >= n_train, so this must stay < n_train.
TARGET_CHUNKS = 6                  # aim for ~this many batches per epoch


def chunk_size_for(n_train: int) -> int:
    """Pick max_data_size so we get ~TARGET_CHUNKS chunks (>=2), always < n_train."""
    size = max(40, n_train // TARGET_CHUNKS)
    return min(size, n_train - 1)


def curriculum_steps(epoch: int) -> int:
    """Curriculum: deepen the latent loop as training progresses
    (COCONUT-inspired staged deepening; depth axis, not latent-count axis)."""
    schedule = [1, 2, TARGET_STEPS]
    stage = min(epoch // EPOCHS_PER_STAGE, len(schedule) - 1)
    return schedule[stage]


def build_estimator(seed: int) -> TabPFNClassifier:
    return TabPFNClassifier(
        n_estimators=1,
        device=DEVICE,
        model_path=MODEL_PATH,
        random_state=seed,
        fit_mode="fit_preprocessors",
    )


def set_thinking(clf: TabPFNClassifier, steps: int, mode: str = "coconut") -> None:
    m = clf.model_
    m._thinking_steps = steps
    m._thinking_mode = mode


def get_drift(clf: TabPFNClassifier):
    return getattr(clf.model_, "_thinking_drift", None)


def evaluate(clf: TabPFNClassifier, X_te, y_te, labels, tag: str) -> dict:
    """Probe query acc/NLL across inference depths (no grad). Returns
    {steps: (acc, nll)} and logs a line per depth."""
    out = {}
    for steps in EVAL_STEPS:
        set_thinking(clf, steps)
        proba = clf.predict_proba(X_te)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_te, pred)
        pc = np.clip(proba, 1e-7, 1.0)
        pc = pc / pc.sum(axis=1, keepdims=True)
        ll = log_loss(y_te, pc, labels=labels)
        drift = get_drift(clf)
        dstr = ""
        if drift:
            dstr = " drift=[" + ", ".join(f"{d[1]:.1f}" for d in drift) + "]"
        log.info("  [%s] steps=%d  acc=%.4f  nll=%.4f%s", tag, steps, acc, ll, dstr)
        out[steps] = (acc, ll)
    set_thinking(clf, 1)
    return out


def finetune_one(name: str, X, y, seed: int) -> dict:
    """Run one (dataset, seed): returns {'pre': {steps:(acc,nll)},
    'post': {steps:(acc,nll)}}."""
    labels = np.unique(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # --- Baseline: fresh estimator, evaluated before any fine-tuning ------------
    base = build_estimator(seed)
    base.fit(X_tr, y_tr)
    log.info("--- [%s seed=%d] BEFORE fine-tune ---", name, seed)
    pre = evaluate(base, X_te, y_te, labels, "pre")

    # --- Fine-tuning estimator (re-uses TabPFN's batched data pipeline) ---------
    est = build_estimator(seed)
    est.fit(X_tr, y_tr)  # initializes models_ / model_
    model = est.model_
    model.train()
    set_thinking(est, 1)
    perf = PerformanceOptions(
        save_peak_memory_factor=None,
        force_recompute_layer=False,
        use_chunkwise_inference=False,
        enable_torch_compile=False,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    max_chunk = chunk_size_for(len(X_tr))

    for epoch in range(EPOCHS):
        steps = curriculum_steps(epoch)
        set_thinking(est, steps)
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
            log.warning("  epoch %2d: 0 chunks (n_train=%d, max_chunk=%d), skip",
                        epoch, len(X_tr), max_chunk)
            continue
        loader = DataLoader(
            ds, batch_size=1, collate_fn=meta_dataset_collator, shuffle=True
        )
        ep_loss, nb, skipped = 0.0, 0, 0
        for batch in loader:
            # skip batches whose query labels aren't covered by context
            ctx_u = torch.unique(
                torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_context])
            )
            qry_u = torch.unique(batch.y_query.reshape(-1))
            if not bool(torch.isin(qry_u, ctx_u, assume_unique=True).all()):
                skipped += 1
                continue
            optim.zero_grad()
            est.fit_from_preprocessed(
                batch.X_context,
                batch.y_context,
                batch.cat_indices,
                batch.configs,
                performance_options=perf,
            )
            # forward on query rows with gradients -> raw logits
            logits_QBEL = est.forward(batch.X_query, return_raw_logits=True)
            Q, B, E, L = logits_QBEL.shape
            logits_BLQ = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
            targets_BQ = (
                batch.y_query.reshape(1, -1).repeat(B * E, 1).to(DEVICE).long()
            )
            loss = _compute_classification_loss(
                logits_BLQ=logits_BLQ, targets_BQ=targets_BQ
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += float(loss.detach().item())
            nb += 1
        log.info(
            "  epoch %2d  steps=%d  batches=%d  skipped=%d  loss=%.4f",
            epoch,
            steps,
            nb,
            skipped,
            ep_loss / max(nb, 1),
        )

    # --- After fine-tune: snapshot trained weights, rebuild predict executor, ----
    # restore (est.fit reloads the checkpoint weights, so we must restore). -------
    model.eval()
    trained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    est.fit(X_tr, y_tr)
    est.model_.load_state_dict(trained_state)
    log.info("--- [%s seed=%d] AFTER fine-tune ---", name, seed)
    post = evaluate(est, X_te, y_te, labels, "post")
    return {"pre": pre, "post": post}


def load_datasets() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """sklearn toy sets (easy, small) + OpenML (larger / harder)."""
    out = []
    for nm, ld in [
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ]:
        d = ld()
        out.append((nm, d.data, d.target))
    # OpenML: larger / harder real tabular datasets
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
    """Aggregate per-seed results into mean±std per (phase, steps)."""
    log.info("##### SUMMARY %s (n_seeds=%d) #####", name, len(runs))
    base_acc = np.mean([r["pre"][1][0] for r in runs])  # pre, steps=1
    for phase in ("pre", "post"):
        for steps in EVAL_STEPS:
            accs = np.array([r[phase][steps][0] for r in runs])
            nlls = np.array([r[phase][steps][1] for r in runs])
            log.info(
                "  %-4s steps=%d  acc=%.4f±%.4f  nll=%.4f±%.4f",
                phase,
                steps,
                accs.mean(),
                accs.std(),
                nlls.mean(),
                nlls.std(),
            )
    # post-vs-pre delta at each depth (paired across seeds)
    for steps in EVAL_STEPS:
        d = np.array(
            [r["post"][steps][0] - r["pre"][steps][0] for r in runs]
        )
        log.info(
            "  Δacc(post-pre) steps=%d: %+.4f±%.4f  (base pre@1=%.4f)",
            steps,
            d.mean(),
            d.std(),
            base_acc,
        )


def main() -> None:
    log.info(
        "COCONUT-inspired looped fine-tune | curriculum=%s over %d epochs | "
        "seeds=%s | ~%d chunks/epoch",
        [curriculum_steps(e) for e in range(EPOCHS)],
        EPOCHS,
        SEEDS,
        TARGET_CHUNKS,
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

