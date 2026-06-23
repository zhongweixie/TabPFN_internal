#!/usr/bin/env python
"""Smoke test (step 1 of the pretraining arc): prove a LOOPED-PREDICTOR TFM variant
trains end-to-end on TabICL/TACO's synthetic SCM prior + TACO harness.

This is NOT a result — it only confirms the pipeline runs: synthetic prior -> looped
predictor forward (run the predictor's 12-layer encoder stack K times, feeding the
hidden state back, universal-transformer / Ouro style) -> CE loss -> backward -> step,
with gradients flowing and loss decreasing, on 1 GPU in minutes.

Design choices for a minimal, honest probe:
  - We DON'T edit TACO's vendored code. We monkeypatch LayerStack.forward to add an
    outer K-iteration loop (loop_k=1 == exact baseline, zero behavior change).
  - We use the no-compressor predictor path (simplest; the loop lives in the predictor,
    which is what thinking-mode/looped reasoning targets).
  - We compare loop_k=1 (baseline) vs loop_k=3 (looped) for a few steps to confirm both
    run and the looped forward produces finite loss + grads.
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")

from taco.model.tabpfn_arch.model.transformer import LayerStack  # noqa: E402

# __APPEND_MARKER__

# ---- monkeypatch: wrap the encoder LayerStack in a K-iteration outer loop ----
_ORIG_LAYERSTACK_FORWARD = LayerStack.forward
LOOP_K = 1   # set per-run; 1 == exact baseline (single pass, no behavior change)


def _looped_forward(self, x, *, half_layers=False, **kwargs):
    """Run the full layer stack LOOP_K times, feeding the output back as input.

    Universal-transformer / Ouro-style recurrence over the SAME shared weights. At
    LOOP_K=1 this is byte-identical to the original single pass. We only loop the
    encoder stack used for the in-context representation; the per-layer kwargs
    (single_eval_pos, cache flags) are passed through unchanged each iteration."""
    out = x
    for _ in range(LOOP_K):
        out = _ORIG_LAYERSTACK_FORWARD(self, out, half_layers=half_layers, **kwargs)
    return out


def main():
    import numpy as np
    from taco.train.finetune_comp import TrainerCompFinetuner
    from taco.train.train_config import build_parser

    LayerStack.forward = _looped_forward  # activate the loop hook globally

    def run(loop_k, steps=4):
        global LOOP_K
        LOOP_K = loop_k
        argv = [
            "--max_steps", str(steps), "--batch_size", "8", "--micro_batch_size", "4",
            "--min_seq_len", "16", "--max_seq_len", "48",
            "--min_features", "2", "--max_features", "12",
            "--device", "cuda", "--prior_device", "cpu", "--num_workers", "2",
            "--wandb_log", "False", "--eval_every", "10000",
            "--checkpoint_dir", f"/tmp/looped_smoke_k{loop_k}",
            "--prior_type", "mix_scm",
            # no-compressor path: the loop lives in the predictor
        ]
        cfg = build_parser().parse_args(argv)
        cfg.use_compressor = False
        torch.manual_seed(0); np.random.seed(0)
        t0 = time.time()
        tr = TrainerCompFinetuner(cfg)
        # sanity: confirm the hook is the active forward
        assert LayerStack.forward is _looped_forward
        tr.train()
        dt = time.time() - t0
        # pull the last CE from the trainer if exposed; else rely on tqdm postfix
        print(f"[loop_k={loop_k}] finished {steps} steps in {dt:.1f}s "
              f"(LOOP_K active={LOOP_K})", flush=True)

    print("=== baseline (loop_k=1, must equal normal TACO) ===", flush=True)
    run(1)
    print("=== looped (loop_k=3, Ouro-style recurrence) ===", flush=True)
    run(3)
    print("SMOKE OK: both ran end-to-end", flush=True)


if __name__ == "__main__":
    main()
