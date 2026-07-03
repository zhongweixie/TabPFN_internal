#!/usr/bin/env python
"""H5: Adaptive Computation Time (ACT) full retrain from scratch.

Adds a learned halting unit to the looped predictor:
  - halt_unit = nn.Linear(emsize, 1)  (no bias, applied to mean test-row hidden state)
  - At each loop step n (1..K_MAX):
      p_n = sigmoid(halt_unit(h_n_test_mean))   # halt probability
      w_n = p_n * remaining_budget              # soft weight for this step's output
      remaining_budget *= (1 - p_n)
  - Weighted output: h_out = sum_n (w_n * h_n)  + remaining_budget * h_K_MAX
  - Ponder cost: N_ponder = sum_n n * p_n  (encourages halting early)
  - Total loss: CE(task) + ACT_LAMBDA * ponder_cost

Trains from scratch for ACT_STEPS with K_MAX loop iterations max.
Lambda search: run with ACT_LAMBDA in {0.01, 0.05, 0.1, 0.2} on separate GPUs.

Usage:
  ACT_LAMBDA=0.01 ACT_K_MAX=6 python train_h5_act.py \\
      --nlayers 12 --max_steps 80000 --checkpoint_dir .../ckpt/h5_act_l001 [TACO args]

Env:
  ACT_LAMBDA   : ponder cost weight (default 0.05)
  ACT_K_MAX    : maximum loop iterations (default 6)
"""
from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import looped_step2 as L

ACT_LAMBDA = float(os.environ.get("ACT_LAMBDA", "0.05"))
ACT_K_MAX = int(os.environ.get("ACT_K_MAX", "6"))
REINJECT_ALPHA = float(os.environ.get("REINJECT_ALPHA", "0.1"))


# ─── ACT halting unit ─────────────────────────────────────────────────────────

class ACTHaltingUnit(nn.Module):
    """Single linear layer: (B, E) -> halt logit (B, 1). No bias — keeps it minimal."""
    def __init__(self, emsize: int = 192):
        super().__init__()
        self.proj = nn.Linear(emsize, 1, bias=False)
        nn.init.zeros_(self.proj.weight)  # start neutral (p_halt ~ 0.5)

    def forward(self, h_mean: torch.Tensor) -> torch.Tensor:
        return self.proj(h_mean)


# ─── ACT looped forward patch ─────────────────────────────────────────────────

# Global registry of ACT state (per-process, single model training)
_ACT_STATE: dict = {
    "ponder_cost": None,    # scalar tensor set during forward
    "n_train": 0,           # set before each forward
    "enabled": False,
    "halting_unit": None,   # ACTHaltingUnit instance
    "k_max": ACT_K_MAX,
}


def install_act_forward():
    """Monkeypatch LayerStack.forward with ACT soft-halting.

    Replaces the fixed-k looped forward with an adaptive one that:
    1. Computes per-step halt probabilities.
    2. Weights outputs softly (differentiable).
    3. Stores ponder cost in _ACT_STATE["ponder_cost"].
    """
    from taco.model.tabpfn_arch.model.transformer import LayerStack

    # Capture original (BEFORE any loop patch, or use L's original)
    if L._ORIG_LAYERSTACK_FWD is None:
        L.install_looped_forward()
    orig = L._ORIG_LAYERSTACK_FWD

    if getattr(LayerStack.forward, "_act_wrapped", False):
        return

    def act_looped(self, x, *, half_layers=False, **kwargs):
        if not _ACT_STATE["enabled"]:
            # Fallback to standard looped forward (for eval)
            k = getattr(self, "_loop_k", L.LOOP_K)
            alpha = getattr(self, "_reinject_alpha", REINJECT_ALPHA)
            h0 = x
            out = x
            for i in range(k):
                out = orig(self, out, half_layers=half_layers, **kwargs)
                if i < k - 1:
                    out = out + alpha * h0
            return out

        # ACT soft-weighted forward — handles both 3D (B,S,E) and 4D (B,S,F,E)
        k_max = _ACT_STATE["k_max"]
        alpha = getattr(self, "_reinject_alpha", REINJECT_ALPHA)
        n_tr  = _ACT_STATE["n_train"]
        halt_unit = _ACT_STATE["halting_unit"]
        B = x.shape[0]

        h0 = x
        h  = x
        hiddens = []  # one entry per step, shape (B, ...)

        for step_i in range(k_max):
            h = orig(self, h, half_layers=half_layers, **kwargs)
            if step_i < k_max - 1:
                h = h + alpha * h0
            hiddens.append(h)

        # ── compute halt probabilities ──────────────────────────────────────
        if halt_unit is not None and n_tr > 0:
            halt_probs = []  # list of (B, 1)
            for h_step in hiddens:
                # Extract test rows and collapse to (B, E) regardless of ndim
                h_test = h_step[:, n_tr:]           # (B, M, ...) — ndim >= 3
                # flatten all middle dims, then mean → (B, E)
                h_test_flat = h_test.reshape(B, -1, h_test.shape[-1]).mean(dim=1)  # (B, E)
                p = torch.sigmoid(halt_unit(h_test_flat))  # (B, 1)
                halt_probs.append(p)

            # ACT soft weights: w_i = remaining * p_i;  remaining *= (1 - p_i)
            weights = []
            remaining = torch.ones(B, 1, device=x.device, dtype=x.dtype)
            for i, p in enumerate(halt_probs):
                if i == k_max - 1:
                    w = remaining  # absorb all remaining at last step
                else:
                    w = remaining * p
                    remaining = remaining * (1.0 - p)
                weights.append(w)            # (B, 1)

            # Weighted output — broadcast w over spatial dims
            weighted_out = torch.zeros_like(hiddens[0])
            for w, h_step in zip(weights, hiddens):
                # w: (B, 1), h_step: (B, S, ...) → expand w to match
                w_broadcast = w.view(B, *([1] * (h_step.ndim - 1)))
                weighted_out = weighted_out + w_broadcast * h_step

            # Ponder cost: E[steps used] = Σ_i (i+1) * p_i * prod_{j<i}(1-p_j)
            # ≈ Σ_i (i+1) * w_i  (weights already encode the stopping probabilities)
            ponder = torch.zeros(B, device=x.device, dtype=x.dtype)
            for step_n, w in enumerate(weights, 1):
                ponder = ponder + step_n * w.squeeze(-1)

            _ACT_STATE["ponder_cost"] = ponder.mean()
            return weighted_out
        else:
            _ACT_STATE["ponder_cost"] = None
            return hiddens[-1]

    act_looped._act_wrapped = True
    LayerStack.forward = act_looped


def enable_act(n_train: int = 0):
    _ACT_STATE["enabled"] = True
    _ACT_STATE["n_train"] = n_train
    _ACT_STATE["ponder_cost"] = None


def disable_act():
    _ACT_STATE["enabled"] = False


# ─── ACT Trainer ──────────────────────────────────────────────────────────────

def make_act_trainer(act_lambda: float, k_max: int):
    from taco.train.finetune_comp import TrainerCompFinetuner

    class ACTTrainer(TrainerCompFinetuner):
        def __init__(self, config):
            super().__init__(config)
            self._act_lambda = act_lambda
            self._k_max = k_max

            # Attach halting unit to raw_model
            emsize = 192
            try:
                emsize = self.raw_model.predictor.transformer_encoder.layers[0].norm1.normalized_shape[0]
            except Exception:
                pass
            halt_unit = ACTHaltingUnit(emsize=emsize).to(self.config.device)
            self.raw_model.act_halting_unit = halt_unit
            _ACT_STATE["halting_unit"] = halt_unit
            _ACT_STATE["k_max"] = k_max

            # Load base checkpoint (curric-60K) BEFORE setting ACT loop so that the
            # pre-trained weights are present before we start ACT fine-tuning.
            # This avoids the ACT collapse that happens when training from scratch.
            base_ckpt = os.environ.get("ACT_BASE_CKPT", "")
            if base_ckpt and os.path.exists(base_ckpt):
                ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
                missing, unexpected = self.raw_model.load_state_dict(
                    ckpt["state_dict"], strict=False)
                print(f"[H5-ACT] Loaded base ckpt {base_ckpt}: "
                      f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            else:
                print(f"[H5-ACT] WARNING: no ACT_BASE_CKPT set, training from scratch", flush=True)

            # Set k_max on all LayerStacks
            L.set_loop_on_model(self.raw_model, k_max, REINJECT_ALPHA)

            # Rebuild optimizer to include halt_unit params
            self.configure_optimizer()
            self.configure_amp()
            print(f"[H5-ACT] Initialized: lambda={act_lambda} k_max={k_max}", flush=True)

        def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches,
                            training_mode: bool = True):
            """Override run_micro_batch to inject ponder cost into the loss
            BEFORE backward — so the computation graph is not freed beforehand."""
            import torch.nn.functional as F

            micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
            seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
            micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)
            micro_X = micro_X.to(self.config.device)
            micro_y = micro_y.to(self.config.device)

            y_train = micro_y[:, :train_size]
            y_test  = micro_y[:, train_size:]
            micro_y = micro_y.detach()

            if training_mode:
                enable_act(n_train=int(train_size))
            else:
                disable_act()

            model = self.model if training_mode else self.raw_model
            with self.amp_ctx:
                pred = model(micro_X, y_train)
                micro_X = micro_X.detach()
                y_train = y_train.detach()
                pred_flat = pred.flatten(end_dim=-2)
                true = y_test.long().flatten()
                ce_loss = F.cross_entropy(pred_flat, true)
                # Combine CE + ACT ponder cost in one backward pass
                if training_mode and _ACT_STATE["ponder_cost"] is not None:
                    total_loss = ce_loss + self._act_lambda * _ACT_STATE["ponder_cost"]
                    ponder_val = float(_ACT_STATE["ponder_cost"].detach())
                else:
                    total_loss = ce_loss
                    ponder_val = 0.0
                y_test = y_test.detach()
                true = true.detach()

            scaled_loss = total_loss / num_micro_batches
            if training_mode and self.model.training:
                self.scaler.scale(scaled_loss).backward()

            disable_act()

            # Stash ponder val for run_batch to pick up (NOT in returned dict —
            # TACO's run_batch only knows {"ce", "accuracy"} and crashes on extras)
            _ACT_STATE["last_ponder"] = _ACT_STATE.get("last_ponder", 0.0) + ponder_val

            with torch.no_grad():
                results = {
                    "ce": ce_loss.detach().float().item() / num_micro_batches,
                    "accuracy": float(
                        (pred_flat.detach().argmax(1) == true).sum()
                    ) / (len(true) * num_micro_batches),
                }
            return results

        def run_batch(self, batch, train: bool = True):
            _ACT_STATE["last_ponder"] = 0.0
            results = super().run_batch(batch, train=train)
            if train and hasattr(self, "wandb_run") and self.wandb_run is not None:
                import wandb
                wandb.log({"act_ponder": _ACT_STATE.get("last_ponder", 0.0)},
                          step=self.curr_step)
            return results

    return ACTTrainer


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    install_act_forward()

    from taco.train.train_config import build_parser
    import looped_train as LT
    from taco.train.finetune_comp import TrainerCompFinetuner

    LT.patch_trainer_for_diagnostics(TrainerCompFinetuner)

    Trainer = make_act_trainer(act_lambda=ACT_LAMBDA, k_max=ACT_K_MAX)

    cfg = build_parser().parse_args()
    cfg.use_compressor = False

    trainer = Trainer(cfg)

    try:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        snap = {k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, type(None)))}
        snap.update({"ACT_LAMBDA": ACT_LAMBDA, "ACT_K_MAX": ACT_K_MAX,
                     "REINJECT_ALPHA": REINJECT_ALPHA})
        json.dump(snap, open(os.path.join(cfg.checkpoint_dir, "config.json"), "w"), indent=1)
    except Exception as e:
        print(f"[H5-ACT] config snapshot skipped: {e}", flush=True)

    print(f"[H5-ACT] Starting: lambda={ACT_LAMBDA} k_max={ACT_K_MAX}", flush=True)
    trainer.train()
    print(f"[H5-ACT] DONE: lambda={ACT_LAMBDA}", flush=True)


if __name__ == "__main__":
    main()
