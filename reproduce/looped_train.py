#!/usr/bin/env python
"""Launcher for ONE step-2 config: installs the re-injection loop patch, then runs
TACO's TrainerCompFinetuner with a held-out eval bank. Driven entirely by env+argv.

Usage (see run_step2.sh):
  LOOP_K=2 REINJECT_ALPHA=0.1 python looped_train.py --nlayers 12 --max_steps 10000 \
      --eval_data_dir .../eval_bank --eval_every 1000 ... (TACO args)
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, "/data/zxiebk/workspace/train/PFN/TabPFN/reproduce")

import looped_step2 as L  # noqa: E402


def main():
    # install the loop BEFORE the model is built
    L.install_looped_forward()
    print(f"[looped_train] LOOP_K={L.LOOP_K} REINJECT_ALPHA={L.REINJECT_ALPHA}", flush=True)

    from taco.train.finetune_comp import TrainerCompFinetuner
    from taco.train.train_config import build_parser

    cfg = build_parser().parse_args()   # consumes the TACO args passed on argv
    cfg.use_compressor = False          # no-compressor path: loop lives in predictor
    trainer = TrainerCompFinetuner(cfg)
    assert L.LOOP_K >= 1
    trainer.train()
    print(f"[looped_train] DONE LOOP_K={L.LOOP_K} nlayers={cfg.nlayers}", flush=True)


if __name__ == "__main__":
    main()