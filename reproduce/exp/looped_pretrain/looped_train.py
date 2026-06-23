#!/usr/bin/env python
"""Launcher for looped-predictor pretraining: installs the re-injection loop patch +
adds wandb diagnostics (grad norm, eval metrics, loop-iteration hidden norms) that
TACO's harness doesn't log by default. Driven entirely by env+argv.

Usage:
  LOOP_K=2 REINJECT_ALPHA=0.1 python looped_train.py --nlayers 12 --max_steps 80000 \
      --eval_data_dir .../eval_bank --eval_every 1000 --wandb_log True ... (TACO args)

Env:
  LOOP_K          : number of recurrence iterations (1=baseline, 2+=looped)
  REINJECT_ALPHA  : residual re-injection weight (default 0.1)
  LOG_DIAG_EVERY  : steps between loop-diagnostic logging (default 100)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import looped_step2 as L  # noqa: E402

LOG_DIAG_EVERY = int(os.environ.get("LOG_DIAG_EVERY", "100"))


def patch_trainer_for_diagnostics(TrainerCls):
    """Monkey-patch the trainer to add wandb logging for grad-norm, eval, and loop diags."""
    import torch
    import torch.nn as nn

    _orig_run_batch = TrainerCls.run_batch
    _orig_evaluate = TrainerCls.evaluate

    def run_batch_with_diag(self, batch, train=True):
        results = _orig_run_batch(self, batch, train=train)
        if train and hasattr(self, "wandb_run") and self.wandb_run is not None:
            import wandb
            # grad norm (after clip — gives effective magnitude)
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.float().norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            extra = {"grad_norm": total_norm}
            # timing (from tqdm postfix if available, otherwise skip)
            if hasattr(self, "_last_prior_time"):
                extra["prior_time"] = self._last_prior_time
            if hasattr(self, "_last_train_time"):
                extra["train_time"] = self._last_train_time
            wandb.log(extra, step=self.curr_step)
        return results

    def evaluate_with_wandb(self):
        _orig_evaluate(self)
        # After the original prints, also push to wandb
        if hasattr(self, "wandb_run") and self.wandb_run is not None:
            import wandb
            # Re-read the last eval result from stdout capture (hack) — or just re-run metric
            # Actually: the original evaluate() sets no attributes. We'll do a simpler patch:
            # just run the eval again on one batch to get the numbers into wandb. BUT that's
            # wasteful. Instead, patch evaluate() to STORE the result.
            pass  # see below — we override evaluate() entirely

    # Full evaluate override that stores + logs
    def evaluate_and_log(self):
        """evaluate() that also logs to wandb."""
        import wandb
        if self.eval_dataloader is None:
            print("No evaluation dataset provided.")
            return
        self.model.eval()
        total_loss = 0.0
        accuracies = []
        N = len(self.eval_dataloader.real_dataset.dataset_files)
        iterator = iter(self.eval_dataloader)
        with torch.no_grad():
            for _ in range(N):
                try:
                    batch = next(iterator)
                    results = self.run_micro_batch(batch, 0, 1, training_mode=False)
                    if results:
                        total_loss += results.get("ce", 0)
                        accuracies.append(results.get("accuracy", 0))
                except Exception:
                    continue
        if accuracies:
            avg_loss = total_loss / len(accuracies)
            avg_acc = sum(accuracies) / len(accuracies)
        else:
            avg_loss, avg_acc = float("nan"), float("nan")
        print(f"Evaluation loss: {avg_loss}", flush=True)
        print(f"Evaluation accuracy at step {self.curr_step}: {avg_acc}", flush=True)
        if hasattr(self, "wandb_run") and self.wandb_run is not None:
            wandb.log({"eval_ce": avg_loss, "eval_accuracy": avg_acc}, step=self.curr_step)
        self.model.train()

    TrainerCls.run_batch = run_batch_with_diag
    TrainerCls.evaluate = evaluate_and_log
    return TrainerCls


def main():
    # install the loop BEFORE the model is built
    L.install_looped_forward()
    print(f"[looped_train] LOOP_K={L.LOOP_K} REINJECT_ALPHA={L.REINJECT_ALPHA}", flush=True)

    from taco.train.finetune_comp import TrainerCompFinetuner
    from taco.train.train_config import build_parser

    patch_trainer_for_diagnostics(TrainerCompFinetuner)

    cfg = build_parser().parse_args()   # consumes the TACO args passed on argv
    cfg.use_compressor = False          # no-compressor path: loop lives in predictor
    trainer = TrainerCompFinetuner(cfg)
    assert L.LOOP_K >= 1
    # Belt-and-suspenders: bind loop_k onto the model instance (not just the global),
    # so the per-instance path the eval uses is exercised identically in training.
    model = getattr(trainer, "raw_model", None) or getattr(trainer, "model", None)
    if model is not None:
        n = L.set_loop_on_model(model, L.LOOP_K, L.REINJECT_ALPHA)
        print(f"[looped_train] bound loop_k={L.LOOP_K} to {n} LayerStack(s)", flush=True)
    # Snapshot the run config next to the checkpoints for reproducibility.
    try:
        ckdir = getattr(cfg, "checkpoint_dir", None)
        if ckdir:
            os.makedirs(ckdir, exist_ok=True)
            import json
            snap = {k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool, type(None)))}
            snap["LOOP_K"] = L.LOOP_K
            snap["REINJECT_ALPHA"] = L.REINJECT_ALPHA
            json.dump(snap, open(os.path.join(ckdir, "config.json"), "w"), indent=1)
    except Exception as e:  # noqa: BLE001
        print(f"[looped_train] config snapshot skipped: {e}", flush=True)
    trainer.train()
    print(f"[looped_train] DONE LOOP_K={L.LOOP_K} nlayers={cfg.nlayers}", flush=True)


if __name__ == "__main__":
    main()