#!/usr/bin/env python
"""H6: Budget-conditional training — the cleanest adaptive-depth approach.

Trains a SINGLE model that works at ANY k value by conditioning on a learned
budget embedding injected at each loop step.

Training:
  - Each batch samples k ~ Uniform(1..K_MAX)
  - Add budget_embedding[remaining_steps] to the re-injection at each loop step
  - Model learns to calibrate its predictions based on "how many more loops I'll do"
  - NO optimizer reset (unlike curriculum) → avoids catastrophic forgetting

Architecture:
  - budget_emb = nn.Embedding(K_MAX + 1, emsize)  # 0..K_MAX budget levels
  - In looped forward at step i (0-indexed, max k-1):
      remaining = k - i - 1  # how many MORE steps after this one
      h = LayerStack(h) + alpha*h0 + budget_emb(remaining)

Inference:
  - Can run at any k ∈ [1..K_MAX] using correct budget embeddings
  - Combined with H1/H2/H3 stopping rules for true adaptive inference

Usage:
  BUDGET_K_MAX=6 python train_h6_budget_cond.py \\
      --nlayers 12 --max_steps 80000 --checkpoint_dir .../ckpt/h6_budget [TACO args]

Env:
  BUDGET_K_MAX        : max loop iterations (default 6)
  BUDGET_SAMPLE_MODE  : 'uniform' or 'biased' (default uniform)
  REINJECT_ALPHA      : re-injection weight (default 0.1)
"""
from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

import looped_step2 as L

BUDGET_K_MAX = int(os.environ.get("BUDGET_K_MAX", "6"))
BUDGET_SAMPLE_MODE = os.environ.get("BUDGET_SAMPLE_MODE", "uniform")
REINJECT_ALPHA = float(os.environ.get("REINJECT_ALPHA", "0.1"))

# ─── Budget embedding module ──────────────────────────────────────────────────

class BudgetEmbedding(nn.Module):
    """Learned embedding for remaining computation budget.

    budget_emb[r] = embedding for "r more steps remaining" where r ∈ [0..k_max].
    """
    def __init__(self, k_max: int, emsize: int = 192):
        super().__init__()
        self.emb = nn.Embedding(k_max + 1, emsize)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, remaining: torch.Tensor) -> torch.Tensor:
        """remaining: (B,) int tensor ∈ [0..k_max] → (B, E)"""
        return self.emb(remaining)


# ─── Budget-conditional looped forward ────────────────────────────────────────

_BUDGET_STATE: dict = {
    "budget_emb": None,     # BudgetEmbedding module
    "current_k": None,      # sampled k for this batch (scalar int)
    "enabled": False,
}


def install_budget_forward():
    """Monkeypatch LayerStack.forward with budget-conditional re-injection."""
    from taco.model.tabpfn_arch.model.transformer import LayerStack

    if L._ORIG_LAYERSTACK_FWD is None:
        L.install_looped_forward()
    orig = L._ORIG_LAYERSTACK_FWD

    if getattr(LayerStack.forward, "_budget_wrapped", False):
        return

    def budget_looped(self, x, *, half_layers=False, **kwargs):
        if not _BUDGET_STATE["enabled"] or _BUDGET_STATE["budget_emb"] is None:
            # Fallback: standard looped forward
            k = getattr(self, "_loop_k", L.LOOP_K)
            alpha = getattr(self, "_reinject_alpha", REINJECT_ALPHA)
            h0 = x
            out = x
            for i in range(k):
                out = orig(self, out, half_layers=half_layers, **kwargs)
                if i < k - 1:
                    out = out + alpha * h0
            return out

        # Budget-conditional forward
        k = _BUDGET_STATE["current_k"]
        alpha = getattr(self, "_reinject_alpha", REINJECT_ALPHA)
        budget_emb = _BUDGET_STATE["budget_emb"]

        h0 = x
        h = x
        B = x.shape[0]
        device = x.device

        # Fix v2: inject budget BEFORE step 0 so k=1 also gets a conditioning signal.
        # budget_emb[k] means "k total steps will be used" — injected into h0 and h.
        def broadcast_emb(vec, target):
            """Broadcast (B, E) vec to match target's ndim."""
            shape = [vec.shape[0]] + [1] * (target.ndim - 2) + [vec.shape[1]]
            return vec.view(*shape).expand_as(target)

        k_tensor = torch.full((B,), k, dtype=torch.long, device=device)
        k_emb = broadcast_emb(budget_emb(k_tensor), h)
        h = h + k_emb      # signal "you will run k steps total"
        h0 = x             # Fix v3: re-injection anchor stays as ORIGINAL tokens (x),
                           # not the budget-conditioned h. Prevents k_emb from leaking
                           # into every step via alpha*h0.

        for step_i in range(k):
            h = orig(self, h, half_layers=half_layers, **kwargs)

            if step_i < k - 1:
                # Re-injection with remaining-steps signal
                remaining = k - step_i - 1
                rem_tensor = torch.full((B,), remaining, dtype=torch.long, device=device)
                rem_emb = broadcast_emb(budget_emb(rem_tensor), h)
                h = h + alpha * h0 + rem_emb

        return h

    budget_looped._budget_wrapped = True
    LayerStack.forward = budget_looped


def enable_budget(k: int):
    _BUDGET_STATE["enabled"] = True
    _BUDGET_STATE["current_k"] = k


def disable_budget():
    _BUDGET_STATE["enabled"] = False


# ─── Budget Trainer ───────────────────────────────────────────────────────────

def make_budget_trainer(k_max: int, sample_mode: str):
    from taco.train.finetune_comp import TrainerCompFinetuner

    class BudgetTrainer(TrainerCompFinetuner):
        def __init__(self, config):
            super().__init__(config)
            self._k_max = k_max
            self._sample_mode = sample_mode
            self._rng = np.random.RandomState(42)

            # Attach budget embedding to raw_model
            emsize = 192
            try:
                emsize = self.raw_model.predictor.transformer_encoder.layers[0]\
                    .norm1.normalized_shape[0]
            except Exception:
                pass

            budget_emb = BudgetEmbedding(k_max=k_max, emsize=emsize).to(self.config.device)
            self.raw_model.budget_embedding = budget_emb
            _BUDGET_STATE["budget_emb"] = budget_emb

            # Load base checkpoint (curric-60K) BEFORE setting the loop/budget,
            # so the pre-trained weights anchor training and avoid divergence.
            base_ckpt = os.environ.get("BUDGET_BASE_CKPT", "")
            if base_ckpt and os.path.exists(base_ckpt):
                ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
                missing, unexpected = self.raw_model.load_state_dict(
                    ckpt["state_dict"], strict=False)
                print(f"[H6-Budget] Loaded base ckpt {base_ckpt}: "
                      f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            else:
                print("[H6-Budget] WARNING: no BUDGET_BASE_CKPT, training from scratch",
                      flush=True)

            # Set initial k on LayerStacks (will be overridden per-batch)
            L.set_loop_on_model(self.raw_model, k_max, REINJECT_ALPHA)

            # Rebuild optimizer to include budget_emb params
            self.configure_optimizer()
            self.configure_amp()

            print(f"[H6-Budget] Initialized: k_max={k_max} sample_mode={sample_mode}",
                  flush=True)

        def run_batch(self, batch, train: bool = True):
            if train:
                # Sample k for this batch
                if self._sample_mode == "uniform":
                    k = int(self._rng.randint(1, self._k_max + 1))
                else:  # biased: favor mid-range k values
                    weights = np.array([1, 2, 3, 3, 2, 1][:self._k_max])
                    weights = weights / weights.sum()
                    k = int(self._rng.choice(range(1, self._k_max + 1), p=weights))
            else:
                # Fix v2: val uses k=k_max WITH budget enabled (same distribution as training).
                # Old code called disable_budget() → val measured model without conditioning
                # signal → val loss was measuring the wrong distribution.
                k = self._k_max

            L.set_loop_on_model(self.raw_model, k, REINJECT_ALPHA)
            enable_budget(k=k)
            results = super().run_batch(batch, train=train)
            disable_budget()
            return results

    return BudgetTrainer


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    install_budget_forward()

    from taco.train.train_config import build_parser
    import looped_train as LT
    from taco.train.finetune_comp import TrainerCompFinetuner

    LT.patch_trainer_for_diagnostics(TrainerCompFinetuner)

    Trainer = make_budget_trainer(k_max=BUDGET_K_MAX, sample_mode=BUDGET_SAMPLE_MODE)

    cfg = build_parser().parse_args()
    cfg.use_compressor = False

    trainer = Trainer(cfg)

    try:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        snap = {k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, type(None)))}
        snap.update({
            "BUDGET_K_MAX": BUDGET_K_MAX,
            "BUDGET_SAMPLE_MODE": BUDGET_SAMPLE_MODE,
            "REINJECT_ALPHA": REINJECT_ALPHA,
        })
        json.dump(snap, open(os.path.join(cfg.checkpoint_dir, "config.json"), "w"),
                  indent=1)
    except Exception as e:
        print(f"[H6-Budget] config snapshot skipped: {e}", flush=True)

    print(f"[H6-Budget] Starting: k_max={BUDGET_K_MAX} mode={BUDGET_SAMPLE_MODE}",
          flush=True)
    trainer.train()
    print(f"[H6-Budget] DONE", flush=True)


if __name__ == "__main__":
    main()
