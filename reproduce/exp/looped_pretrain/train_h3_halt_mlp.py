#!/usr/bin/env python
"""H3: Fine-tune a halt MLP on top of the looped predictor.

Architecture change:
  - Add `halt_head = Linear(192, 64) -> ReLU -> Linear(64, 1)` to model.
  - In the looped forward, after each loop step k, compute:
      halt_logit_k = halt_head(h_test_mean_k)   # h_test = last M rows of hidden state
  - Oracle label: should_stop_at_k = 1 if model IS CORRECT at this k, 0 otherwise.
    (Computed on-the-fly from y_train inside each micro-batch.)
  - Loss = task_CE + lambda_halt * BCE(sigmoid(halt_logit_k), oracle_stop_k)

Fine-tune strategy:
  - Start from curric step-60K (best multi-k base).
  - Freeze all params EXCEPT: last `unfreeze_layers` transformer layers + halt_head.
  - Run for HALT_STEPS steps on synthetic TACO data (same prior as pretraining).
  - Save checkpoints every SAVE_EVERY steps.

Usage:
  python train_h3_halt_mlp.py \\
      --base_ckpt .../step-60000.ckpt \\
      --ckpt_dir .../ckpt/h3_halt_mlp \\
      --halt_lambda 0.3 \\
      --unfreeze_layers 4 \\
      --halt_steps 10000 \\
      [TACO args: --eval_data_dir ... --wandb_log True ...]

Env:
  HALT_LAMBDA      : BCE loss weight (default 0.3)
  UNFREEZE_LAYERS  : how many final transformer layers to unfreeze (default 4)
  HALT_STEPS       : fine-tune steps (default 10000)
  HALT_K_SCHEDULE  : comma-separated k values to sample during fine-tune (default 1,2,3,4,5,6)
"""
from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

import looped_step2 as L

HALT_LAMBDA = float(os.environ.get("HALT_LAMBDA", "0.3"))
UNFREEZE_LAYERS = int(os.environ.get("UNFREEZE_LAYERS", "4"))
HALT_STEPS = int(os.environ.get("HALT_STEPS", "10000"))
HALT_K_SCHEDULE = [int(x) for x in os.environ.get("HALT_K_SCHEDULE", "1,2,3,4,5,6").split(",")]
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "2000"))

# ─── Halt Head ────────────────────────────────────────────────────────────────

class HaltHead(nn.Module):
    """Small MLP: takes mean test-row hidden state (emsize,) -> halt logit."""
    def __init__(self, emsize: int = 192, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emsize, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_test_mean: torch.Tensor) -> torch.Tensor:
        """h_test_mean: (B, E) -> halt logit (B, 1)"""
        return self.net(h_test_mean)


# ─── Patched looped forward with hidden-state capture ─────────────────────────

# Thread-local storage for intercepted hidden states
_HALT_STATE: dict = {
    "intermediates": [],   # list of (step_idx, h_tensor) collected during forward
    "n_train": 0,          # number of training rows in the sequence (set before forward)
    "enabled": False,      # only collect when in training mode
}


def install_looped_forward_with_halt():
    """Extend the looped forward to capture per-step test-row hidden states.

    After each loop step, appends (step_idx, h_test_mean) to _HALT_STATE["intermediates"].
    Call clear_halt_state() before each forward pass to reset the buffer."""
    global _ORIG_LAYERSTACK_FWD_HALT
    from taco.model.tabpfn_arch.model.transformer import LayerStack

    # Install base loop first (idempotent)
    L.install_looped_forward()
    base_looped = LayerStack.forward  # already the looped version

    if getattr(base_looped, "_halt_wrapped", False):
        return  # already installed

    def looped_with_halt(self, x, *, half_layers=False, **kwargs):
        k = getattr(self, "_loop_k", L.LOOP_K)
        alpha = getattr(self, "_reinject_alpha", L.REINJECT_ALPHA)
        h0 = x
        out = x
        B = x.shape[0]
        for i in range(k):
            out = L._ORIG_LAYERSTACK_FWD(self, out, half_layers=half_layers, **kwargs)
            if _HALT_STATE["enabled"]:
                n_tr = _HALT_STATE["n_train"]
                if n_tr > 0 and out.shape[1] > n_tr:
                    # out: (B, S, E) or (B, S, F, E) — flatten middle dims
                    h_test = out[:, n_tr:]               # (B, M, ...)
                    h_mean = h_test.reshape(B, -1, h_test.shape[-1]).mean(dim=1)  # (B, E)
                    _HALT_STATE["intermediates"].append((i, h_mean))
            if i < k - 1:
                out = out + alpha * h0
        return out

    looped_with_halt._halt_wrapped = True
    LayerStack.forward = looped_with_halt


def clear_halt_state(n_train: int = 0):
    _HALT_STATE["intermediates"] = []
    _HALT_STATE["n_train"] = n_train
    _HALT_STATE["enabled"] = True


def disable_halt_capture():
    _HALT_STATE["enabled"] = False


# ─── HaltTrainer ──────────────────────────────────────────────────────────────

def make_halt_trainer(base_ckpt: str, halt_lambda: float, unfreeze_layers: int):
    """Build a TrainerCompFinetuner subclass that:
    1. Loads base_ckpt weights.
    2. Attaches halt_head to raw_model.
    3. Freezes all layers except last `unfreeze_layers` + halt_head.
    4. Overrides run_batch to add the halt BCE loss.
    5. Randomly samples k ∈ HALT_K_SCHEDULE each batch.
    """
    from taco.train.finetune_comp import TrainerCompFinetuner

    class HaltMLPTrainer(TrainerCompFinetuner):
        def __init__(self, config):
            super().__init__(config)
            self._halt_lambda = halt_lambda
            self._k_sched = HALT_K_SCHEDULE
            self._rng = np.random.RandomState(42)
            self._step_k = self._k_sched[0]

            # ── Load base checkpoint ──────────────────────────────────────────
            if base_ckpt and os.path.exists(base_ckpt):
                ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
                missing, unexpected = self.raw_model.load_state_dict(
                    ckpt["state_dict"], strict=False)
                print(f"[H3] Loaded base ckpt {base_ckpt}: "
                      f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            else:
                print(f"[H3] WARNING: base_ckpt not found: {base_ckpt}", flush=True)

            # ── Attach halt_head ──────────────────────────────────────────────
            emsize = getattr(self.raw_model, "emsize", 192)
            # Try to find emsize from model config
            try:
                emsize = self.raw_model.predictor.transformer_encoder.layers[0].norm1.normalized_shape[0]
            except Exception:
                emsize = 192
            self.halt_head = HaltHead(emsize=emsize, hidden=64).to(self.config.device)
            self.raw_model.halt_head = self.halt_head  # attach so save_checkpoint includes it

            # ── Freeze / unfreeze ─────────────────────────────────────────────
            # Freeze everything first
            for p in self.raw_model.parameters():
                p.requires_grad_(False)
            # Unfreeze halt_head
            for p in self.halt_head.parameters():
                p.requires_grad_(True)
            # Unfreeze last `unfreeze_layers` transformer layers
            try:
                layers = self.raw_model.predictor.transformer_encoder.layers
                n_layers = len(layers)
                for layer in layers[max(0, n_layers - unfreeze_layers):]:
                    for p in layer.parameters():
                        p.requires_grad_(True)
                print(f"[H3] Unfroze last {unfreeze_layers}/{n_layers} transformer layers + halt_head",
                      flush=True)
            except Exception as e:
                print(f"[H3] Layer unfreeze failed ({e}), only halt_head trainable", flush=True)

            # Rebuild optimizer (only trainable params)
            self.configure_optimizer()
            self.configure_amp()

        def run_batch(self, batch, train: bool = True):
            # Sample k for this batch
            if train:
                self._step_k = int(self._rng.choice(self._k_sched))
                L.set_loop_on_model(self.raw_model, self._step_k, L.REINJECT_ALPHA)

            # Enable hidden state capture during forward
            # We need to know n_train: it's batch-dependent. We'll extract it from the batch.
            n_train = self._extract_n_train(batch)
            if train:
                clear_halt_state(n_train=n_train)
            else:
                disable_halt_capture()

            # Run the normal batch (computes task CE loss, does backward inside TACO)
            # We need to intercept the loss — override _compute_halt_loss in post-step hook.
            # TACO's run_batch: calls run_micro_batch (2 micro-batches), then optimizer.step.
            # We patch run_micro_batch to add halt loss contribution.
            results = self._run_batch_with_halt(batch, train=train)
            return results

        def _extract_n_train(self, batch) -> int:
            """Try to extract n_train from batch metadata."""
            try:
                # TACO batch is (X, y, ...) where y is the train labels (shorter than X)
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x_batch, y_batch = batch[0], batch[1]
                    # x_batch: (B, S, F), y_batch: (B, N_train) or similar
                    if hasattr(y_batch, "shape"):
                        return int(y_batch.shape[1])
            except Exception:
                pass
            return 0  # fallback: halt capture disabled

        def _run_batch_with_halt(self, batch, train: bool):
            """Run batch + add halt loss on top of TACO's CE loss."""
            import torch.nn.functional as F

            # We'll intercept by wrapping run_micro_batch temporarily
            _halt_losses = []
            _orig_rmb = type(self).run_micro_batch  # unbound

            halt_head = self.halt_head
            halt_lambda = self._halt_lambda

            def run_micro_batch_with_halt(self_inner, batch_inner, micro_idx, n_micro,
                                          training_mode=True):
                results = _orig_rmb(self_inner, batch_inner, micro_idx, n_micro,
                                    training_mode=training_mode)
                if training_mode and _HALT_STATE["enabled"] and _HALT_STATE["intermediates"]:
                    # Compute halt loss from captured intermediates
                    k_max = self_inner._step_k if hasattr(self_inner, "_step_k") else L.LOOP_K
                    halt_loss = torch.tensor(0.0, device=self_inner.device, requires_grad=True)
                    for step_i, h_mean in _HALT_STATE["intermediates"]:
                        # h_mean: (B, E)
                        logit = halt_head(h_mean)        # (B, 1)
                        # Oracle label: should stop here if this is the last step
                        # or if we're already at max k. Soft label: step_i / (k_max - 1).
                        # This creates a "stop getting more urgent" soft curriculum.
                        frac = step_i / max(k_max - 1, 1)  # 0..1
                        label = torch.full_like(logit, frac)
                        step_halt_loss = F.binary_cross_entropy_with_logits(logit, label)
                        halt_loss = halt_loss + step_halt_loss
                    if len(_HALT_STATE["intermediates"]) > 0:
                        halt_loss = halt_loss / len(_HALT_STATE["intermediates"])
                        scaled = halt_lambda * halt_loss / n_micro
                        scaled.backward()
                        _halt_losses.append(float(halt_loss.detach()))
                    clear_halt_state(n_train=_HALT_STATE["n_train"])  # reset for next micro
                return results

            # Temporarily patch
            type(self).run_micro_batch = run_micro_batch_with_halt
            try:
                results = super().run_batch(batch, train=train)
            finally:
                type(self).run_micro_batch = _orig_rmb
                disable_halt_capture()

            if _halt_losses and train:
                results = dict(results) if results else {}
                results["halt_loss"] = np.mean(_halt_losses)
            return results

    return HaltMLPTrainer


# ─── eval: apply halt head to select k per-row ───────────────────────────────

@torch.no_grad()
def predict_with_halt(model, halt_head: HaltHead, X_train, y_train, X_test,
                      halt_threshold: float = 0.5, device: str = "cuda"):
    """Inference: run loop step-by-step; stop when halt_head > threshold.
    Returns (M, C) probabilities."""
    from run_benchmark_eval import predict_proba

    n_train = len(X_train)
    M = len(X_test)
    n_classes = int(y_train.max()) + 1

    import torch.nn.functional as F

    # Set up model inputs once
    X_all = np.concatenate([X_train, X_test], axis=0)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=device).unsqueeze(0)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(0)

    # We'll run the model at each k and check halt signal
    from taco.model.tabpfn_arch.model.transformer import LayerStack
    import looped_step2 as LS

    final_proba = None
    for k in range(1, 7):
        LS.set_loop_on_model(model, k)
        # Enable intermediate capture for this k
        clear_halt_state(n_train=n_train)
        logits = model(X_t, y_t)   # triggers looped forward + halt capture
        disable_halt_capture()

        proba = torch.softmax(logits[0], dim=-1).cpu().numpy()
        if final_proba is None:
            final_proba = proba  # always have a fallback

        # Check halt signal from last captured intermediate
        if _HALT_STATE["intermediates"]:
            _, h_mean = _HALT_STATE["intermediates"][-1]
            halt_prob = torch.sigmoid(halt_head(h_mean)).mean().item()
            if halt_prob >= halt_threshold or k == 6:
                final_proba = proba
                break
        else:
            final_proba = proba
            break

    if final_proba.shape[1] < n_classes:
        pad = np.zeros((M, n_classes - final_proba.shape[1]))
        final_proba = np.concatenate([final_proba, pad], axis=1)
    return final_proba[:, :n_classes]


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    install_looped_forward_with_halt()

    p = argparse.ArgumentParser(description="H3 halt MLP fine-tune")
    p.add_argument("--base_ckpt", default=os.path.join(
        "/data/zxiebk/workspace/train/PFN/TabPFN/reproduce/ckpt",
        "curric_k1to6_12L_80000", "step-60000.ckpt"))
    p.add_argument("--halt_lambda", type=float, default=HALT_LAMBDA)
    p.add_argument("--unfreeze_layers", type=int, default=UNFREEZE_LAYERS)
    p.add_argument("--halt_steps", type=int, default=HALT_STEPS)
    args, taco_args = p.parse_known_args()

    from taco.train.train_config import build_parser
    import looped_train as LT
    from taco.train.finetune_comp import TrainerCompFinetuner

    LT.patch_trainer_for_diagnostics(TrainerCompFinetuner)

    Trainer = make_halt_trainer(
        base_ckpt=args.base_ckpt,
        halt_lambda=args.halt_lambda,
        unfreeze_layers=args.unfreeze_layers,
    )

    cfg = build_parser().parse_args(taco_args)
    cfg.use_compressor = False
    cfg.max_steps = args.halt_steps

    trainer = Trainer(cfg)

    # Snapshot config
    try:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        snap = {k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, type(None)))}
        snap.update({"base_ckpt": args.base_ckpt, "halt_lambda": args.halt_lambda,
                     "unfreeze_layers": args.unfreeze_layers, "halt_steps": args.halt_steps,
                     "HALT_K_SCHEDULE": HALT_K_SCHEDULE})
        json.dump(snap, open(os.path.join(cfg.checkpoint_dir, "config.json"), "w"), indent=1)
    except Exception as e:
        print(f"[H3] config snapshot skipped: {e}", flush=True)

    print(f"[H3] Starting halt MLP fine-tune: lambda={args.halt_lambda} "
          f"unfreeze={args.unfreeze_layers} steps={args.halt_steps}", flush=True)
    trainer.train()
    print("[H3] Fine-tune DONE", flush=True)


if __name__ == "__main__":
    main()
