#!/usr/bin/env python
"""Variable-loop-k CURRICULUM training (COCONUT-style) for TabPFN looped predictor.

COCONUT's curriculum = progressively replace text reasoning steps with latent steps,
resetting the optimizer at each stage switch (from iCoT / Deng 2024). Our mapping:
progressively INCREASE loop_k (depth-recurrence count) in stages, rebuilding the
optimizer+scheduler (clearing Adam momentum) at each switch.

Stages (default): loop_k = 1 -> 2 -> 3 -> 4 -> 5 -> 6, equal step budget each.
At each switch:
  - set_loop_on_model(model, new_k)         # retarget the single encoder LayerStack
  - configure_optimizer()                   # rebuild AdamW (clears momentum) + scheduler
  - configure_amp()                         # fresh GradScaler
  - record stage/loop_k for checkpoint resume

Goal: a SINGLE model that works across loop_k=1..6 (all seen + calibrated), which is the
prerequisite for any adaptive (binary-classifier) depth selection — unlike the fixed-k
models where off-k inference degenerates (Q1) and maxConf failed.

Implementation: subclass TrainerCompFinetuner, override train() with stage logic inserted
at the per-step boundary. TACO source is NOT modified.

Verified facts this relies on (empirically checked 2026-06-23):
  - model has exactly ONE LayerStack (predictor.transformer_encoder); loop applies only there
  - configure_optimizer() rebuilds BOTH optimizer+scheduler from raw_model+config, re-callable
  - configure_amp() rebuilds scaler; train() is step-based (curr_step = step+1)
"""

from __future__ import annotations

import os
import sys
import json

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import looped_step2 as L  # noqa: E402

# curriculum schedule (overridable by env)
LOOP_K_SCHEDULE = [int(x) for x in os.environ.get("LOOPK_SCHED", "1,2,3,4,5,6").split(",")]
REINJECT_ALPHA = float(os.environ.get("REINJECT_ALPHA", "0.1"))

# __APPEND_MARKER__


def stage_for_step(step, total_steps, n_stages):
    """Equal-width stages over [0, total_steps). Returns stage index in [0, n_stages)."""
    per = max(1, total_steps // n_stages)
    return min(step // per, n_stages - 1)


def make_curriculum_trainer():
    """Subclass TrainerCompFinetuner: at each step, if the curriculum stage changed,
    switch loop_k and rebuild optimizer+scheduler+scaler (COCONUT/iCoT optimizer reset)."""
    from taco.train.finetune_comp import TrainerCompFinetuner

    class CurriculumTrainer(TrainerCompFinetuner):
        def __init__(self, config):
            super().__init__(config)
            self._sched = LOOP_K_SCHEDULE
            self._cur_stage = getattr(self, "_cur_stage", -1)
            self._cur_loop_k = getattr(self, "_cur_loop_k", None)
            # apply the stage matching the (possibly resumed) curr_step BEFORE training
            self._apply_stage(stage_for_step(self.curr_step, self.config.max_steps, len(self._sched)),
                              rebuild=False)

        def _apply_stage(self, stage, rebuild=True):
            if stage == self._cur_stage:
                return
            new_k = self._sched[stage]
            n = L.set_loop_on_model(self.raw_model, new_k, REINJECT_ALPHA)
            self._cur_stage = stage
            self._cur_loop_k = new_k
            if rebuild:
                # COCONUT/iCoT: reset optimizer momentum + scheduler + scaler at stage switch
                self.configure_optimizer()
                self.configure_amp()
            if self.master_process:
                print(f"[curriculum] step={self.curr_step} -> stage={stage} loop_k={new_k} "
                      f"(bound {n} LayerStack, rebuilt={rebuild})", flush=True)

        def run_batch(self, batch, train=True):
            # stage check at the per-step boundary (run_batch is called once per step)
            if train:
                st = stage_for_step(self.curr_step, self.config.max_steps, len(self._sched))
                self._apply_stage(st, rebuild=True)
            return super().run_batch(batch, train=train)

        def save_checkpoint(self, name):
            super().save_checkpoint(name)
            # record curriculum state alongside (for resume)
            try:
                import torch
                p = os.path.join(self.config.checkpoint_dir, name)
                ck = torch.load(p, map_location="cpu", weights_only=False)
                ck["curriculum_stage"] = self._cur_stage
                ck["curriculum_loop_k"] = self._cur_loop_k
                torch.save(ck, p)
            except Exception as e:  # noqa: BLE001
                print(f"[curriculum] ckpt stage-tag skipped: {e}", flush=True)

    return CurriculumTrainer


def main():
    L.install_looped_forward()
    from taco.train.train_config import build_parser
    # reuse the diagnostics patch (wandb eval/grad-norm) from looped_train
    import looped_train as LT
    from taco.train.finetune_comp import TrainerCompFinetuner
    LT.patch_trainer_for_diagnostics(TrainerCompFinetuner)

    Trainer = make_curriculum_trainer()
    cfg = build_parser().parse_args()
    cfg.use_compressor = False
    print(f"[curriculum] schedule loop_k={LOOP_K_SCHEDULE} alpha={REINJECT_ALPHA} "
          f"max_steps={cfg.max_steps}", flush=True)
    trainer = Trainer(cfg)
    # config snapshot
    try:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        snap = {k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, type(None)))}
        snap["LOOP_K_SCHEDULE"] = LOOP_K_SCHEDULE
        snap["REINJECT_ALPHA"] = REINJECT_ALPHA
        json.dump(snap, open(os.path.join(cfg.checkpoint_dir, "config.json"), "w"), indent=1)
    except Exception as e:  # noqa: BLE001
        print(f"[curriculum] config snapshot skipped: {e}", flush=True)
    trainer.train()
    print(f"[curriculum] DONE schedule={LOOP_K_SCHEDULE}", flush=True)


if __name__ == "__main__":
    main()
