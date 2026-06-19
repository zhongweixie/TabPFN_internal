# Adaptive Ensemble Depth on TabPFN v2.5 — Ceiling Gate (NO-GO)

**Date:** 2026-06-19
**Status:** DIRECTION CLOSED — falsified at the cheap ceiling gate, no head trained.
**Script:** `reproduce/run_adaptive_ceiling.py` · **Log:** `reproduce/adaptive_ceiling_run.out`
**Robustness addendum (post-review):** `reproduce/run_robustness.py` §A — see end.

## Motivation

After the learned **row-selection** direction closed (see `SPARSE_ROWS_INDEXER_1A.md`,
`SPARSE_ROWS_GLOBAL_1B.md`), we reopened "can FlashMemory-V4 help TabPFN?" via a
*structurally different* mechanism. FlashMemory's actual selling point is not raw accuracy
but a **Pareto improvement**: equal compute → better accuracy, or equal accuracy → less
compute (its headline is 13.5% KV cache at equal accuracy). The TabPFN analogue of "spend
compute where it helps" is **adaptive ensemble depth**: instead of running the same
`n_estimators` for every test row, give *hard* rows more ensemble members and *easy* rows
fewer, at a **fixed average budget**. This predicts *how much compute a row needs*, not
*which rows matter* — so it sidesteps the "row importance doesn't generalize" wall that
killed row-selection.

## Decisive question (ceiling-first, same discipline as the LOO gate)

> At a fixed average budget B (mean estimators-per-row), can an **oracle** that allocates
> ensemble members optimally across rows beat **uniform** allocation?

If even a label-aware oracle can't move the Pareto curve, no learned difficulty head can —
NO-GO, cheaply, before training anything.

## Method (one fit gives every budget for free)

`TabPFNClassifier.predict_raw_logits(X)` returns per-**ensemble-member** logits `(K, M, C)`,
de-permuted and class-aligned to `clf.classes_` (de-permutation at `classifier.py:1538`,
before the dim-0 mean). So a single `K_MAX`-member fit yields, for every test row, its
prediction at budget `k = 1..K_MAX` as the running mean of its first `k` members' softmax
probs — no extra inference cost. `K_MAX=32`, `N_train=1000`, `N_test=300`, seeds `{0,1}`,
budgets `B ∈ {1,2,4,8,16}`, 4 OpenML datasets spanning clean→noisy.

Three allocations per budget, all at the same total `B·M` members:
- **uniform** — every row gets `B` members (deployable baseline).
- **oracle (greedy)** — start each row at 1, repeatedly give one member to the row with the
  largest marginal true-loss reduction. Uses the true label → upper bound, not deployable.
- **unsup** — deployable label-free proxy: allocate by member disagreement (predictive
  entropy + per-class member spread). Previews what a learned head could capture.

**Snoop check (the decisive one): cross-fit oracle.** The plain oracle decides each row's
budget on the *same* members it is scored on — so it can cherry-pick whichever `k` happens
to land closest to the true label on this particular noise draw. The cross-fit oracle splits
the `K` members in half: decide the per-row budget on the **decide** half, evaluate on the
disjoint **eval** half. If the gain survives, some rows genuinely converge slower (real
adaptive-depth structure). If it collapses, the plain oracle was overfitting ensemble noise.

## Results

Δ = strategy − uniform, at the same average budget. (acc up / logloss down = better.)

| dataset | plain ORACLE Δacc / Δll (best B) | **cross-fit oracle Δacc / Δll** | unsup proxy Δacc |
|---|---|---|---|
| breast_cancer (clean) | +0.003 / −0.008 | ~0.000 / +0.000 | ~0 |
| qsar-biodeg | +0.005 / −0.013 | +0.010 / ~0.000 | ~0 |
| phoneme | +0.013 / −0.017 | **−0.005 / +0.005** | ~0 |
| electricity (noisy) | **+0.032 / −0.037** | **−0.010 / +0.007** | +0.005 |

## Interpretation — NO-GO

1. **The plain-oracle headroom is a snooping artifact.** Its most positive result
   (electricity +3.2% acc, −0.037 logloss at B=16) **goes negative under cross-fit**
   (−0.010 acc, +0.007 logloss). phoneme does the same (+1.3% → −0.005). The plain oracle
   was picking, per row, whichever budget landed closest to the truth *on the members it was
   graded on* — overfitting ensemble noise, not discovering slow-converging rows.

2. **No stable per-row convergence structure exists.** Once the budget decision cannot see
   the evaluation members, there is nothing to exploit. TabPFN's ensemble members (which
   differ by preprocessing / feature-shift / class-permutation) converge to essentially the
   **same per-row answer at the same rate** — they do not disagree in a row-specific,
   persistent way that selective extra averaging would fix.

3. **The deployable proxy confirms it independently.** The label-free disagreement signal
   captures ≤0.5% even of the (inflated) plain-oracle gap — consistent with there being no
   real signal to capture.

This is the same lesson as row-selection, now confirmed for compute allocation: per-row
behavior on a frozen TabPFN is **collective and already near-converged**, leaving no stable
per-row signal for a small decoupled head to predict.

**Cost to reach this verdict:** ~30 lines of allocation logic + one ~12s run. No difficulty
head trained, no GPU-days spent — exactly what the ceiling-first gate is for.

## Bottom line for FlashMemory-on-TabPFN

Three FlashMemory→TabPFN mappings now falsified on frozen v2.5: per-query row selection,
global cross-dataset indexer, adaptive ensemble depth. The common wall: **post-hoc
lightweight modules can't exploit per-row structure that TabPFN's pretrained forward pass
has already collapsed.** FlashMemory's gains depend on LLM KV-cache per-token redundancy;
TabPFN's in-context memory lacks the analogous per-row slack. The remaining untested angles
(per-row calibration head, OOD/abstention head) are post-hoc too and expected marginal;
genuinely changing the verdict would require intervening in **pre-training**, not bolting a
module onto v2.5.

## Robustness addendum (answering the design/adversarial review)

Multi-subagent review judged the NO-GO "sound but narrow": clean datasets, no label
noise, K=32, single 50/50 cross-fit split. The top-plausibility attack: "adaptive depth
is a denoising play — it only helps under label noise / large K, the regime not tested."
`reproduce/run_robustness.py` §A re-ran the snoop-free cross-fit oracle under
train label-flip ∈ {0, 10, 25%}, **K_MAX=64**, 3 seeds, AND a split-ratio sweep
(50/50 and 75/25 — to rule out "halving K is too harsh").

Result: the snoop-free Δacc over uniform stays at the **noise floor everywhere** —
across all noise levels, both splits, all budgets. Examples at 25% label flip:
electricity Δacc −0.001..+0.004, qsar −0.011..+0.003, phoneme −0.011..+0.002. **Label
noise did NOT create adaptive-depth headroom**, and the 75/25 split did not recover any
effect. This directly refutes both the "wrong regime / denoising" and "split too harsh"
attacks. Mechanistic reason confirmed: more ensemble members average over the *stochastic
ensemble axis* (preprocessing/permutation), not the data axis — so they cannot recover a
flipped training label the way row-removal could, and there is no per-row "needs-more-
members" structure for budgeting to exploit, clean or noisy.

**Verdict: NO-GO is robust**, not a clean/small-K/single-split artifact.
