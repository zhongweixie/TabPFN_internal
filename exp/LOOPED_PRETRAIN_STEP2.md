# Looped/Recurrent Predictor at Pretraining — Step-2 Go/No-Go (NO-GO)

**Date:** 2026-06-19
**Status:** NO-GO at 10K-step trend gate. Recurrence gives no gain; equal compute is
better spent on depth. Honest caveat: 10K << paper's 80K (undertrained trend, not final).
**Scripts:** `reproduce/looped_step2.py`, `looped_train.py`, `run_step2.sh`
**Logs:** `reproduce/step2_{c1_base,c2_loop,c3_deep}.out`

## Question

Step 1 proved the looped-predictor pipeline runs. Step 2 asks the real question: does a
**looped/recurrent predictor** (run the 12-layer encoder stack K times, shared weights,
universal-transformer / Ouro style) help in-context tabular pretraining — at **matched
parameters** AND against a **matched-compute depth control**?

## Design (built to the design red-team's spec, to make a NO-GO safe)

Three confounds were neutralized before running:
- **Naive-looping collapse** (our earlier thinking-mode arc saw Ouro-style recurrence
  collapse 0.983→0.096): fixed with **input re-injection** `h ← LayerStack(h); h ← h +
  α·h₀` (α=0.1), the minimal Ouro/COCONUT stabilizer. Forward-selftest: loop_k=1 is
  byte-identical to baseline; loop_k=2 activates and differs.
- **Compute confound** (loop_k=2 @12L = 24 layer-applications vs baseline's 12): added a
  **matched-compute control** c3 = loop_k=1 @24L (same 24 applications, 2× params).
- **Fair A/B**: same `--np_seed/--torch_seed` → identical synthetic train stream; a
  **fixed held-out eval bank** (20 SCM CSVs at a held-out seed) → identical eval. Verified:
  c1 and c2 have identical step-0 eval (same init).

| config | recurrence | layer-applications (compute) | params |
|---|---|---|---|
| c1 base | loop_k=1 @12L | 12 | 1× |
| c2 loop | loop_k=2 @12L (re-injection) | 24 | 1× |
| c3 deep | loop_k=1 @24L | 24 | 2× |

**GO criterion:** c2 ≥ c3 AND c2 > c1 (recurrence must beat equal-compute depth, not just
spend more FLOPs). 10K steps each, GPUs 0/1/5.

## Results (held-out eval bank; late-window mean = steps 5000–9000, 5 points)

| config | acc (late mean) | CE (late mean) | best acc |
|---|---|---|---|
| c1 base (k1@12L) | 0.612 ± 0.010 | 0.978 | 0.630 |
| **c2 loop (k2@12L)** | **0.602 ± 0.005** | 0.977 | 0.616 |
| **c3 deep (k1@24L)** | **0.620 ± 0.006** | **0.961** | 0.629 |

## Verdict — NO-GO

Both halves of the GO criterion fail:
1. **c2 ≈ c1** (0.602 vs 0.612; looping is if anything slightly *worse*) → doubling the
   baseline's compute via recurrence buys nothing.
2. **c2 < c3** (0.602 vs 0.620, and CE 0.977 vs 0.961) → at the *same* 24-layer-application
   compute, plain depth beats recurrence. The extra compute is better spent on more
   distinct layers than on re-running shared ones.

The matched-compute control (c3) is what makes this conclusive: c2-vs-c1 alone (≈ tie)
could be misread as "harmless"; c3 reveals recurrence actively *wastes* compute that depth
would have used.

## Why this is credible (and not an artifact)

- **Not naive collapse**: re-injection worked — all three curves converge normally, c2
  doesn't collapse, it simply has no advantage.
- **Not a compute confound**: c3 controls compute; depth wins at equal compute.
- **Not a ceiling effect**: eval acc ≈ 0.62 (not saturated) — there is headroom the
  recurrence failed to capture, so "no difference" isn't "no room".
- **Fair**: identical seed/train-stream/eval-bank; c1≡c2 at init.

## Scope / caveat

10K steps << the paper's 80K — formally a *trend gate*, not a final verdict. But the trend
is consistent and monotone: c2 tracks c1 the whole way and stays below c3, with no sign of
late-training crossover. A longer run is unlikely to reverse a gap this stable; if revisited,
the cheapest decisive extension is c2 vs c3 to ~40K steps.

## Consistency with the broader program

Same root cause as the FlashMemory 4/4 NO-GO: **in-context tabular prediction is a
single-pass "read train rows → emit logits" computation with no iterative-refinement
structure for recurrence to exploit** — unlike LLM multi-step reasoning where looping
(Ouro/COCONUT) pays off. Adding *depth* (more distinct layers) helps; re-running shared
layers does not. This was the cheapest pretraining-level question on the roadmap; it is
answered NO-GO before committing to the 20-day path-C scale.
