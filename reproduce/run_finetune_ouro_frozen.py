#!/usr/bin/env python
"""Branch B rescue: freeze-encoder + early-stopping Ouro fine-tune.

The bug-fixed Ouro run still degraded post@1 below baseline on every dataset.
The hypothesis (from review): the damage is small-data fine-tuning ERODING the
pretrained single-pass representation (esp. the input encoder), NOT the looped
recurrence itself. This script tests that hypothesis causally by:
  1. FREEZING the input encoder/embedders (feature_group_embedder, target_embedder,
     feature_positional_embedding_embeddings) so the pretrained representation is
     protected, training only blocks + decoder + thinking rows + gate.
  2. EARLY STOPPING on a held-out validation split (restore best weights), so we
     don't train into overfitting.

Prediction: if the erosion hypothesis is right, post@1 should stop collapsing
(stay near the pre@1 baseline). Whether depth then helps is the secondary question.

Reuses ouro_loss / ensure_gate / evaluate / build_estimator etc. from
run_finetune_ouro.py to keep the method identical except for the two changes.
"""

from __future__ import annotations

import logging
import sys
import time
from functools import partial

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import run_finetune_ouro as O
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
        logging.FileHandler("finetune_ouro_frozen_results.log", mode="w"),
    ],
)
log = logging.getLogger("ouro_frozen")
O.log = log  # route O.evaluate's logging here too

FROZEN_PREFIXES = (
    "feature_group_embedder",
    "target_embedder",
    "feature_positional_embedding_embeddings",
)
PATIENCE = 3
VAL_SPLIT = 0.25

# __APPEND_MARKER__


def val_loss(est, model, X_val_batches) -> float:
    """Mean per-step CE over preprocessed val batches (no grad)."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in X_val_batches:
            est.fit_from_preprocessed(
                batch.X_context, batch.y_context, batch.cat_indices, batch.configs,
                performance_options=O.PerformanceOptions(
                    save_peak_memory_factor=None, force_recompute_layer=False,
                    use_chunkwise_inference=False, enable_torch_compile=False,
                ),
            )
            _ = est.forward(batch.X_query, return_raw_logits=True)
            tgt = batch.y_query.reshape(1, -1).to(O.DEVICE).long()
            loss, _ = O.ouro_loss(model._thinking_step_logits, tgt)
            tot += float(loss.item()); n += 1
    return tot / max(n, 1)


def finetune_one(name: str, X, y, seed: int) -> dict:
    labels = np.unique(y)
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    # carve a validation slice out of the training data for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=VAL_SPLIT, random_state=seed, stratify=y_tr_full
    )

    base = O.build_estimator(seed)
    base.fit(X_tr_full, y_tr_full)
    log.info("--- [%s seed=%d] BEFORE fine-tune ---", name, seed)
    pre = O.evaluate(base, X_te, y_te, labels, "pre")

    est = O.build_estimator(seed)
    est.fit(X_tr, y_tr)
    model = est.model_
    model.train()
    O.set_thinking(est, O.T_MAX)
    gate = O.ensure_gate(est, seed)
    model._thinking_collect_step_logits = True

    # FREEZE the input encoder / embedders.
    n_frozen = 0
    for n, p in model.named_parameters():
        if n.startswith(FROZEN_PREFIXES):
            p.requires_grad_(False)
            n_frozen += 1
    trainable = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "_ouro_step_gate" not in n
    ]
    optim = torch.optim.AdamW(
        [{"params": trainable, "lr": O.LR},
         {"params": [gate], "lr": O.GATE_LR}],
        weight_decay=0.01,
    )
    perf = PerformanceOptions(
        save_peak_memory_factor=None, force_recompute_layer=False,
        use_chunkwise_inference=False, enable_torch_compile=False,
    )
    max_chunk = O.chunk_size_for(len(X_tr))
    log.info("  froze %d encoder params; %d trainable tensors", n_frozen, len(trainable))

    best_vl, best_state, since = float("inf"), None, 0
    for epoch in range(O.EPOCHS):
        epoch_seed = 100 * seed + epoch
        splitter = partial(train_test_split, test_size=O.QUERY_SPLIT,
                           random_state=epoch_seed)
        ds = get_preprocessed_dataset_chunks(
            calling_instance=est, X_raw=X_tr, y_raw=y_tr, split_fn=splitter,
            max_data_size=max_chunk, model_type="classifier",
            equal_split_size=False, data_shuffle_seed=epoch_seed,
            preprocessing_random_state=epoch_seed,
        )
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=1, collate_fn=meta_dataset_collator,
                            shuffle=True)
        ep_loss, nb = 0.0, 0
        for batch in loader:
            ctx_u = torch.unique(
                torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_context])
            )
            if not bool(torch.isin(torch.unique(batch.y_query.reshape(-1)),
                                   ctx_u, assume_unique=True).all()):
                continue
            optim.zero_grad()
            est.fit_from_preprocessed(
                batch.X_context, batch.y_context, batch.cat_indices, batch.configs,
                performance_options=perf,
            )
            _ = est.forward(batch.X_query, return_raw_logits=True)
            tgt = batch.y_query.reshape(1, -1).to(O.DEVICE).long()
            loss, _ = O.ouro_loss(model._thinking_step_logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += float(loss.detach().item()); nb += 1

        # validation for early stopping (build val batches once per epoch)
        val_ds = get_preprocessed_dataset_chunks(
            calling_instance=est, X_raw=X_val, y_raw=y_val,
            split_fn=partial(train_test_split, test_size=0.5, random_state=epoch_seed),
            max_data_size=None, model_type="classifier", equal_split_size=False,
            data_shuffle_seed=epoch_seed, preprocessing_random_state=epoch_seed,
        )
        vbatches = list(DataLoader(val_ds, batch_size=1,
                                   collate_fn=meta_dataset_collator))
        vl = val_loss(est, model, vbatches) if vbatches else float("nan")
        gate_str = "[" + ", ".join(f"{torch.tanh(g).item():+.2f}" for g in gate) + "]"
        log.info("  epoch %2d  tr=%.4f  val=%.4f  gate=%s",
                 epoch, ep_loss / max(nb, 1), vl, gate_str)
        if vl < best_vl - 1e-4:
            best_vl, since = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                log.info("  early stop @ epoch %d (best val=%.4f)", epoch, best_vl)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); model._thinking_collect_step_logits = False
    trained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gate_snapshot = gate.detach().clone()
    est.fit(X_tr_full, y_tr_full)
    est.model_.load_state_dict(trained_state, strict=False)
    est.model_._ouro_step_gate = torch.nn.Parameter(gate_snapshot)
    log.info("--- [%s seed=%d] AFTER fine-tune (frozen-enc + early-stop) ---",
             name, seed)
    post = O.evaluate(est, X_te, y_te, labels, "post")
    return {"pre": pre, "post": post}


def main() -> None:
    log.info(
        "Ouro RESCUE: freeze encoder + early-stop | T_max=%d | patience=%d | seeds=%s",
        O.T_MAX, PATIENCE, O.SEEDS,
    )
    for name, X, y in O.load_datasets():
        log.info("=" * 72)
        log.info("DATASET %s  shape=%s  classes=%d", name, X.shape, len(np.unique(y)))
        runs = []
        t0 = time.time()
        for seed in O.SEEDS:
            runs.append(finetune_one(name, X, y, seed))
        O.aggregate_and_log(name, runs)
        log.info("%s done in %.1fs", name, time.time() - t0)
    log.info("ALL DONE")


if __name__ == "__main__":
    main()
