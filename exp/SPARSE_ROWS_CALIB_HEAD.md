# Per-Row Calibration Head on TabPFN v2.5 — Gate (NO-GO)

**Date:** 2026-06-19
**Status:** DIRECTION CLOSED — strongest-form negative (objective itself wants a global scalar).
**Script:** `reproduce/run_calib_head.py` · **Log:** `reproduce/calib_head_run.out`

## Motivation

The 4th and last FlashMemory→TabPFN angle on frozen v2.5. A tiny decoupled MLP on the
frozen backbone predicts a **per-row softmax temperature** `T_i` from label-free
uncertainty features. Structurally distinct from the three falsified directions: it's
**post-hoc on the final logits**, and temperature scaling `T>0` **preserves argmax**, so
**accuracy is mathematically unchanged**. This is purely a calibration (NLL / ECE) play —
the only question is whether a *per-row* temperature beats a *single global* one.

## Honest gate (snoop-free, train/eval split)

- Fit on **VAL**, evaluate on **disjoint TEST** (never seen by either calibrator):
  - (0) uncalibrated `T=1`
  - (1) **global temperature** `T* = argmin val-NLL` — the standard strong baseline
- Candidate (2): **conditional** `T_i = head(features_i)`, head trained on val NLL, eval test.
  Features = frozen, label-free per-row signals only: predictive entropy, max-prob, margin,
  ensemble-member disagreement (entropy/max-prob std across members), logit gap.
- **GO** if conditional-T beats global-T on held-out NLL **and** ECE. **NO-GO** if ≈ equal.

Head design: `T_i = T_global · exp(Δ_i)`, `Δ_i = MLP(features)`, zero-init so at start
`T_i ≡ T_global` (identity to the strong baseline) — it can only move if a per-row signal
helps. `K_MAX=32`, `N_train=1000`, `N_val=N_test=300`, seeds `{0,1}`, 4 OpenML datasets.

## Results (held-out test; cond − global, lower = better)

| dataset | NLL global → cond (Δ) | ECE global → cond (Δ) | T_global | per-row T std |
|---|---|---|---|---|
| breast_cancer | 0.0857 → 0.0861 (+0.0003) | 0.0302 → 0.0304 (+0.0002) | 1.02 | 0.00 |
| phoneme | 0.3394 → 0.3406 (+0.0012) | 0.0546 → 0.0565 (+0.0019) | 0.92 | 0.00 |
| electricity | 0.4054 → 0.4057 (+0.0003) | 0.0591 → 0.0581 (−0.0011) | 0.92 | 0.00 |
| qsar-biodeg | 0.2797 → 0.2784 (−0.0013) | 0.0481 → 0.0421 (−0.0060) | 1.10 | 0.00 |

Conditional ≈ global everywhere (noise-level deltas, mixed sign). Note `T_global ≈ 1` —
TabPFN v2.5 is already near-calibrated, so even the global temperature does little. Global-T
itself is sometimes worse than uncalibrated on test (qsar) — ordinary val→test variance.

## The decisive diagnostic (rules out the dead-head failure mode)

`per-row T std = 0.0000` could mean two very different things: a **dead/underfit head** (the
ReLU dead-zone bug we hit in the row-selection work), or a **genuine optimum at a constant**.
A targeted diagnostic on electricity (the noisiest, best shot at per-row signal) separates them:

1. **Features vary** — per-row std nonzero (entropy 0.193, logit-gap 1.102, …). Real input.
2. **Head has capacity** — fed a *synthetic* per-row target correlated with the entropy
   feature, it fits it perfectly: MSE 0.0017, pred-std 1.502 = target-std 1.502. The head is
   alive and expressive.
3. **Yet on the real NLL objective it collapses to a constant** — per-row T std `0.0000`
   even at **2000 epochs / lr 1e-2**. The optimizer is free to vary T per row and *chooses
   not to*.

So this is the strongest form of NO-GO: not "we failed to learn a per-row temperature" but
**the calibration objective itself wants a global scalar** — the NLL-optimal per-row
temperature *is* the global one. There is no per-row calibration structure to exploit.

## Bottom line — FlashMemory→TabPFN: 4/4 post-hoc mappings falsified

| direction | mechanism | verdict |
|---|---|---|
| row selection (per-query) | which train rows to attend to | NO-GO (`SPARSE_ROWS_INDEXER_1A.md`) |
| global cross-dataset indexer | shared row-selection rule | NO-GO (`SPARSE_ROWS_GLOBAL_1B.md`) |
| adaptive ensemble depth | how much compute per row | NO-GO (`SPARSE_ROWS_ADAPTIVE_DEPTH.md`) |
| **per-row calibration head** | per-row temperature | **NO-GO (this report)** |

The wall is consistent and now well-evidenced: **a frozen pretrained TabPFN forward pass has
already collapsed per-row structure**, so decoupled lightweight heads have nothing per-row to
exploit — the opposite of the LLM KV-cache per-token redundancy FlashMemory relies on. The
only untested angle (D: OOD / abstention head) is a *selective-prediction utility* play, not
an accuracy or calibration gain. Changing this verdict would require intervening in
**pre-training**, not bolting a module onto v2.5.

## Robustness addendum (answering the design/adversarial review)

Review's main calibration attack: "scalar temperature is the *weakest* per-row intervention
(monotone, argmax-preserving); a richer head (vector/per-class scaling) might find structure
scalar-T cannot — and scalar-T collapsing to a constant does not prove no per-row structure."
Also flagged: wd=1e-3 + tight clamp could bias the scalar head toward the constant.

`reproduce/run_robustness.py` §B answers this with a **vector (per-class) temperature head**
`T_{i,c} = Tg_c · exp(delta_{i,c})` (C× the capacity of scalar T), zero-init to a **per-class
global vector** baseline (itself stronger than a single scalar), **wd=0**, wider clamp
(±3), 64-wide MLP, 800 epochs, 3 seeds.

Result (held-out test, vector − global-vector):

| dataset | NLL Δ | ECE Δ | per-row T std |
|---|---|---|---|
| breast_cancer | +0.0001 | −0.0007 | 0.106 |
| phoneme | +0.0012 | +0.0044 | 0.017 |
| electricity | +0.0027 | −0.0002 | 0.018 |
| qsar-biodeg | +0.0083 | +0.0066 | 0.013 |

Key contrast with the scalar head: the vector head does **not** collapse (per-row T std up
to 0.106) — it genuinely explores per-row solutions. **Yet it still fails to beat the global
per-class vector on held-out test**: NLL is worse everywhere, ECE mixed/worse on average.
The richer intervention found per-row structure *on validation* that **does not generalize**
— it overfits, the opposite of a real signal. This refutes the "scalar-T too weak" and
"reg biases toward null" attacks: with no weight decay and far more capacity, per-row
calibration still loses to a global (per-class) constant on unseen data.

**Verdict: calibration NO-GO holds beyond the scalar case.**
