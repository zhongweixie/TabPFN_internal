# HCA-Style Global Fallback for TabPFN — Analysis & Three Paths (A/B/C)

**Date:** 2026-06-19
**Status:** ANALYSIS / ROADMAP — not yet attempted. Parked for a future arc.
**Context:** Closes out the "why did 4/4 post-hoc FlashMemory mappings fail?" thread with a
root-cause, and proposes the one structurally-coherent way forward (path C).

## The question

Can we give TabPFN a "global compressed fallback layer" (analogous to DeepSeek-V4's HCA),
then do sparsity on top — i.e. **learned context *compression* rather than *selection*?**

## What HCA actually does in FlashMemory (three conditions, all required)

HCA is not merely a "backstop". It makes sparse selection *safe* by simultaneously:
1. **A lossy global summary is always present** (128:1 compression of the whole history),
   so even if LSA prunes every fine-grained CSA chunk, coarse semantics survive → selection
   "can be wrong without dying".
2. **HCA and fine-grained CSA coexist *from pretraining*** — the model learned "read coarse
   via HCA, read fine via CSA". That division of labor is a **pretrained inductive bias**,
   not something bolted on afterward.
3. Therefore LSA only sparsifies the *prunable* path (CSA); the risk is absorbed by HCA.

**Load-bearing consequence: HCA's fallback ability is a product of pretraining. It cannot be
attached post-hoc.** This is the crux of the whole question.

## TabPFN's current state: it already has an HCA-shaped thing, pointed the wrong way

- **Thinking rows (64)**: pretrained constant rows prepended to every dataset; test rows
  attend to them (`AddThinkingRows`, `tabpfn_v2_5.py:76`). They are global, compressed,
  always-present — **structurally the closest thing to HCA**.
- **But they are dataset-INDEPENDENT constants** — they do not vary with *your* training set.
  They provide the model's generic "scratch pad", not a lossy summary of *your* training rows.
  So they **cannot** serve as the fallback for "I pruned the training rows, lean on this" —
  they don't encode the training rows at all.

So the proposal, stated precisely: **upgrade thinking rows from dataset-independent constants
into a dataset-dependent lossy compressed summary, then do sparsity on the raw training rows.**
That is the faithful mapping of HCA.

## Three paths, increasing cost; A and B predicted NO-GO, C is the only coherent one

### Path A — freeze TabPFN, train a compressor that emits "summary rows" into the thinking-row slot
- This is exactly **TACO's compressor**, but TACO proves it must be **co-trained end-to-end
  with the predictor** (≈20 days, 8×H100). Reason = condition 2 above: a frozen TabPFN never
  learned "how to read a compressed summary"; its attention weights were trained to read raw
  rows. Feed it a compressed latent and it doesn't know what to do with it.
- Our calibration-head / RowIndexer experiments are a **preview** of this path: frozen backbone
  + bolted-on small module → can't learn anything that generalizes. A compressor is *harder*
  than an indexer (it must reconstruct information, not just score), so the frozen path is even
  more doomed. **Predicted NO-GO, with our own empirical support.**

### Path B — freeze + per-query sparsity on raw rows + global summary fallback
- If the fallback uses thinking rows → they don't encode your data → can't catch the fall.
- If the fallback is freshly built → degenerates into Path A.
- And the "per-query sparse row selection" half is **exactly the cell we falsified 4/4**.
- So B = (dead fallback) + (dead sparsity). **NO-GO.**

### Path C — pretrain a "dual-path TFM" from scratch: compressed-summary path + fine-grained sparse path coexisting (like HCA + CSA)
- The **only architecturally self-consistent** path: fallback ability comes from pretraining,
  sparse risk is absorbed, all three HCA conditions reproduced.
- But it is **not "add a layer to TabPFN" — it defines a new TFM pretraining objective.** Cost
  = TACO tier (20 days, 8×H100+). TACO already occupies the nearest niche ("global continuous
  compression"); our increment would have to be "compressed summary + *learned sparsity* on top"
  — one layer more than TACO, more expensive, and the payoff (memory saving) may not beat TACO's
  plain end-to-end compression.

## Root-cause conclusion (the clean ending for the negative-result writeup)

The intuition is **architecturally correct**: an HCA-style fallback *is* what makes sparsity
viable, and it's exactly what TabPFN lacks; thinking rows are the closest shell. But there is an
unavoidable constraint: **fallback ability is a pretraining product and cannot be attached
post-hoc.**
- Want it cheap (freeze TabPFN) → fallback is ineffective or degenerates into a compressor →
  hits the same "frozen + bolt-on can't learn" wall as our four NO-GOs.
- Want it correct (pretrain dual paths) → TACO-tier new-TFM project, niche partly taken.

This does not overturn the prior verdict — it **strengthens** the main thesis: for FlashMemory's
spirit to hold on TabPFN, the only real entry point is **pretraining intervention**, which is
precisely the exit all our negative results point to. The unifying root cause of why every
post-hoc attempt failed: **TabPFN lacks a pretrained lossy global fallback, and a fallback cannot
be added after the fact.**

## Next-arc roadmap (user-directed 2026-06-19, deferred)

Before attempting A/B/C, revisit test-time compute with the dual-path lens, then go to pretraining:
1. **Thinking-mode + looped-transformer / COCONUT / Ouro** — re-explore latent-iteration / looped
   reasoning on TabPFN (the earlier thinking-mode arc concluded NO-GO for naive looping; revisit
   with the compression/fallback framing). See exp/THINKING_MODE_*.md.
2. **Pretraining following TACO's recipe** — path C, using TACO's training recipe as the template.
   Resources on disk:
   - `/home/zxiebk/workspace/train/PFN/TACO` — TACO paper (`paper_taco.md`) + code (compressor +
     predictor, TabPFNv2 arch, ~80K steps, 20d×8 H100; up to 94x faster, 97% less memory,
     "1% compression ~lossless" on TabArena).
   - `/home/zxiebk/workspace/train/PFN/tabicl` — TabICL paper + code (retrieval-augmented TFM).

## Pretraining data + recipe — SOLVED (2026-06-19, verified by reading both repos)

The blocker for ALL pretraining-based paths (path C *and* a real thinking-mode/looped/COCONUT/Ouro
attempt, which also need pretraining to make the model *learn to use* the mechanism) was "where do
the data + recipe come from, since TabPFN's official pretraining is unreproducible." Both pieces
turn out to be on-disk, runnable, and TabPFN-weight-independent:

**DATA — TabICL's synthetic SCM prior generator (the most valuable reusable asset):**
- 100% synthetic, sampled from Structural Causal Models (MLP-SCM + Tree-SCM mix, 70/30). NO real
  data needed, NO TabPFN weights needed. Code: `tabicl/src/tabicl/prior/` (`_dataset.py` →
  `PriorDataset(IterableDataset)`, `_mlp_scm.py`, `_tree_scm.py`, `_genload.py` for save/load).
- Runs STANDALONE: `python -m tabicl.prior --save_dir ... --num_batches ... --prior_type mix_scm`,
  or import directly: `from tabicl.prior._dataset import PriorDataset` → yields (X, y, n_features,
  train_sizes) batches for any custom training loop.
- **TACO does not reinvent this — it directly reuses TabICL v1's prior** ("We adopt the synthetic
  prior released with TabICLv1"). So this generator is the shared data foundation for both.
- Caveat: pre-generated data is NOT shipped (generator only); TabICL *v2* pretraining code is
  unreleased (only v1 training scripts) — but the prior generator itself is fully present.

**RECIPE — TACO's training harness is complete and the dual-tower skeleton matches path C:**
- Entry: `TACO/src/taco/train/finetune_comp.py` + `scripts/train_stage1_taco_random.sh`. Full DDP,
  grad-accum, bf16, checkpointing. Trains FROM RANDOM INIT (no TabPFN ckpt reused).
- Architecture = compressor + predictor, BOTH TabPFNv2 arch (12 layers, emsize 192, 6 heads, ~7M
  each), wired via a ResidualMLP bridge; compressor emits a low-rank latent context (K≪N rows) fed
  to the predictor. Multi-rate compression sampling {1,2,4,8,16}%. Loss = end-to-end cross-entropy.
- Recipe: 80K steps, global batch 1024, micro-batch 16, AdamW, lr 1e-4 cosine + 2% warmup, grad
  clip 1.0, ~20d × 8 H100. (TabICL v2's own recipe: Muon, 550K steps, ~24.5 H100-days, 3-stage
  curriculum — heavier; TACO is the closer template for us.)
- **Reuse vs rewrite for our variants:** reuse the prior generator + training harness (DDP/ckpt/
  accum/loss scaffold) AS-IS; REWRITE only the model's `__init__`/forward to inject our mechanism
  (thinking-mode loop, or compressed-summary + sparse fine-grained dual path).

**Compute ladder (start cheap, gate before scaling):**
1. Smoke: ~100 steps, 1 GPU, minutes — confirm the modified architecture trains end-to-end.
2. Small signal: ~10K steps, 2 GPUs, 2–5 days — see if the mechanism learns / trend emerges.
3. Near-paper: 80K steps, 20d × 8 H100 — production-grade, only if (2) is promising.

**Implication:** we do NOT need to reproduce TabPFN's official prior. Standing on TabICL (data) +
TACO (recipe/skeleton) and modifying only the architecture is the realistic on-ramp. The 20-day
figure is the *ceiling*, not the *entry cost* — a meaningful go/no-go is reachable at the ~10K-step
/ 2-GPU rung.
