#!/usr/bin/env python
"""Phase 0 diagnostic probe: COCONUT/Ouro-style "thinking mode" on TabPFN v2.5.

We reuse the (weight-shared) v2.5 transformer block stack for multiple passes at
inference. v2.5 already prepends learnable "thinking rows" (AddThinkingRows) which
play the role of COCONUT's latent thought tokens; the train+test rows play the role
of the fixed "question" context.

Two recurrence modes (set on the underlying model):
  - "coconut": carry refined thinking rows forward, reset the data context to its
    initial embedding each pass (latent-only feedback).
  - "ouro": keep the full hidden state and re-apply the whole stack (full recurrence).

GOAL of this probe is DIAGNOSTIC, not to win accuracy for free. We expect the
pretrained (fixed-depth) weights NOT to give a free monotonic lift. What we want:
  1. confirm the loop runs and is byte-identical at n_steps=1,
  2. measure hidden-state DRIFT ||h^t - h^{t-1}|| vs step -> does the recurrence
     approach a fixed point (drift decays) or diverge/explode?,
  3. see whether any acc/NLL signal moves with steps.
These decide whether Phase 1 (looped fine-tune) needs per-step norm / a learnable
step scalar to stay stable.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("thinking_probe_results.log", mode="w"),
    ],
)
log = logging.getLogger("thinking_probe")

DEVICE = "cuda"
MODEL_PATH = (
    "/home/zxiebk/workspace/model/tabpfn_2_5/"
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)
STEPS = [1, 2, 3, 4, 6, 8]
MODES = ["coconut", "ouro"]
SEED = 0


def build_clf() -> TabPFNClassifier:
    # fit_preprocessors -> plain (no-cache) forward path, which is where the
    # thinking loop is active. n_estimators=1 keeps a single clean forward.
    return TabPFNClassifier(
        n_estimators=1,
        device=DEVICE,
        model_path=MODEL_PATH,
        random_state=SEED,
        fit_mode="fit_preprocessors",
    )


def set_thinking(clf: TabPFNClassifier, steps: int, mode: str) -> None:
    model = clf.model_
    model._thinking_steps = steps
    model._thinking_mode = mode
    if hasattr(model, "_thinking_drift"):
        del model._thinking_drift


def get_drift(clf: TabPFNClassifier):
    return getattr(clf.model_, "_thinking_drift", None)


def run_dataset(name: str, X, y) -> None:
    log.info("=" * 70)
    log.info("DATASET %s  shape=%s  classes=%d", name, X.shape, len(np.unique(y)))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    for mode in MODES:
        log.info("-" * 50)
        log.info("MODE = %s", mode)
        base_acc = None
        for steps in STEPS:
            clf = build_clf()
            clf.fit(X_tr, y_tr)
            set_thinking(clf, steps, mode)
            t0 = time.time()
            proba = clf.predict_proba(X_te)
            dt = time.time() - t0
            pred = np.argmax(proba, axis=1)
            acc = accuracy_score(y_te, pred)
            # guard against degenerate proba for log_loss
            proba_c = np.clip(proba, 1e-7, 1.0)
            proba_c = proba_c / proba_c.sum(axis=1, keepdims=True)
            ll = log_loss(y_te, proba_c, labels=np.unique(y))
            if steps == 1:
                base_acc = acc
            drift = get_drift(clf)
            drift_str = ""
            if drift:
                think = [f"{d[1]:.3f}" for d in drift]
                drift_str = " think_drift=[" + ", ".join(think) + "]"
            dacc = "" if base_acc is None else f"  dAcc={acc - base_acc:+.4f}"
            log.info(
                "steps=%d  acc=%.4f%s  nll=%.4f  t=%.2fs%s",
                steps,
                acc,
                dacc,
                ll,
                dt,
                drift_str,
            )


def main() -> None:
    log.info("Thinking-mode probe | model=%s | device=%s", MODEL_PATH, DEVICE)
    datasets = [
        ("wine", *_xy(load_wine)),
        ("breast_cancer", *_xy(load_breast_cancer)),
        ("digits", *_xy(load_digits)),
    ]
    for name, X, y in datasets:
        run_dataset(name, X, y)
    log.info("DONE")


def _xy(loader):
    d = loader()
    return d.data, d.target


if __name__ == "__main__":
    main()
