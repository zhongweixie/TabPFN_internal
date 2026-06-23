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

# mutable box for passing the true pre-clip grad norm out of clip_grad_norm_ (see below)
_GRAD_NORM_BOX = {"val": None}


def patch_trainer_for_diagnostics(TrainerCls):
    """Monkey-patch the trainer to add wandb logging for grad-norm, eval, and loop diags."""
    import torch
    import torch.nn as nn

    _orig_run_batch = TrainerCls.run_batch
    _orig_evaluate = TrainerCls.evaluate

    # Capture the TRUE pre-clip grad norm. TACO's run_batch zeroes grads (set_to_none=True)
    # before returning, so reading p.grad afterward always sees None -> norm 0. Instead we
    # wrap nn.utils.clip_grad_norm_ (which run_batch calls when gradient_clipping > 0 and
    # whose return value — the total norm before clipping — it discards) and stash the value.
    if not getattr(nn.utils.clip_grad_norm_, "_diag_wrapped", False):
        _real_clip = nn.utils.clip_grad_norm_

        def _clip_capture(parameters, max_norm, *a, **kw):
            total = _real_clip(parameters, max_norm, *a, **kw)
            try:
                _GRAD_NORM_BOX["val"] = float(total)
            except Exception:  # noqa: BLE001
                _GRAD_NORM_BOX["val"] = None
            return total

        _clip_capture._diag_wrapped = True
        nn.utils.clip_grad_norm_ = _clip_capture

    def run_batch_with_diag(self, batch, train=True):
        _GRAD_NORM_BOX["val"] = None  # reset; populated during _orig_run_batch's clip step
        results = _orig_run_batch(self, batch, train=train)
        if train and hasattr(self, "wandb_run") and self.wandb_run is not None:
            import wandb
            # true pre-clip grad norm captured from clip_grad_norm_; None if clipping is off
            total_norm = _GRAD_NORM_BOX["val"]
            extra = {}
            if total_norm is not None:
                extra["grad_norm"] = total_norm
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