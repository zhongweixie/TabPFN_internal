#!/usr/bin/env python
"""Step 2 go/no-go: does a LOOPED/recurrent predictor help in-context tabular
pretraining, at matched params AND with a matched-COMPUTE control?

Built to the red-team's spec (avoids the 4 confounds that would make a NO-GO unsafe):
  * RE-INJECTION: naive output->input looping collapsed in the earlier thinking-mode
    arc (Ouro 0.983->0.096). We re-inject the original input each iteration
    (h <- LayerStack(h); then h <- h + ALPHA*h0), the minimal Ouro/COCONUT fix.
  * MATCHED-COMPUTE CONTROL: loop_k=2 @12L = 24 layer-applications. We also run
    loop_k=1 @24L (same compute/depth, 2x params). GO requires looped >= deep.
  * FAIR A/B: same --np_seed/--torch_seed => identical synthetic train stream across
    configs; a FIXED pre-generated eval bank => identical held-out eval.
  * HONEST SCOPE: 10K steps << paper's 80K; this is a trend gate, not a final verdict.

Sub-commands:
  gen-eval   : generate a fixed bank of held-out SCM datasets as CSV (target column).
  forward-selftest : verify the re-injection looped forward runs + loop_k=1 == baseline.
(The actual training runs are launched by run_step2.sh, which calls TACO's
finetune_comp.py with the loop monkeypatch installed via PYTHONSTARTUP-style import.)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/zxiebk/workspace/train/PFN/TACO/src")

# __APPEND_MARKER__


def gen_eval(out_dir, n_datasets=20, seq_len=600, min_feat=5, max_feat=40, seed=123):
    """Generate a FIXED bank of held-out SCM datasets as CSV (features + 'target').

    Uses the same prior as training but a DIFFERENT, fixed seed so the eval set is
    disjoint-in-distribution-sample from training and identical across all configs.
    The eval loader does an internal 80/20 split per CSV."""
    import os
    import torch
    from taco.prior.dataset import PriorDataset

    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)
    ds = PriorDataset(
        batch_size=1, batch_size_per_gp=1,
        min_features=min_feat, max_features=max_feat, max_classes=10,
        min_seq_len=seq_len, max_seq_len=seq_len + 1,
        log_seq_len=False, seq_len_per_gp=False,
        min_train_size=0.1, max_train_size=0.9, replay_small=False,
        prior_type="mix_scm", device="cpu", n_jobs=1,
    )
    it = iter(ds)
    written = 0
    while written < n_datasets:
        X, y, d, seq_lens, train_sizes = next(it)
        # X: (B,S,F) or nested; take first group, trim to real feature count
        Xb = X[0] if X.dim() == 3 else X
        yb = y[0] if y.dim() == 2 else y
        F = int(d[0]) if hasattr(d, "__len__") else int(d)
        Xb = Xb[:, :F].cpu().numpy().astype(np.float32)
        yb = yb.cpu().numpy().astype(np.float32)
        if len(np.unique(yb)) < 2:
            continue                                   # skip degenerate (single-class)
        df = pd.DataFrame(Xb, columns=[f"f{j}" for j in range(F)])
        df["target"] = yb
        df.to_csv(os.path.join(out_dir, f"eval_{written:03d}.csv"), index=False)
        written += 1
    print(f"wrote {written} eval CSVs to {out_dir} (seq_len={seq_len}, feat={min_feat}-{max_feat})")


# ---- the re-injection looped forward (installed onto LayerStack.forward) ----
# These globals are read by the monkeypatch; set via env by run_step2.sh.
import os as _os  # noqa: E402

LOOP_K = int(_os.environ.get("LOOP_K", "1"))
REINJECT_ALPHA = float(_os.environ.get("REINJECT_ALPHA", "0.1"))


_ORIG_LAYERSTACK_FWD = None  # captured once, the TRUE original forward


def install_looped_forward():
    """Monkeypatch LayerStack.forward: run the stack LOOP_K times with input
    re-injection. LOOP_K=1 reduces to the exact original (the re-inject term only
    fires between iterations, of which there are none).

    IDEMPOTENT: captures the true original exactly once (guards against re-patching
    a patched function, which would nest the loop). Reads LOOP_K/REINJECT_ALPHA from
    module globals at call-time, so changing them between model loads takes effect
    without re-patching."""
    global _ORIG_LAYERSTACK_FWD
    from taco.model.tabpfn_arch.model.transformer import LayerStack
    if _ORIG_LAYERSTACK_FWD is None:
        _ORIG_LAYERSTACK_FWD = LayerStack.forward
    orig = _ORIG_LAYERSTACK_FWD

    def looped(self, x, *, half_layers=False, **kwargs):
        h0 = x
        out = x
        for i in range(LOOP_K):
            out = orig(self, out, half_layers=half_layers, **kwargs)
            if i < LOOP_K - 1:
                out = out + REINJECT_ALPHA * h0   # Ouro/COCONUT-style re-injection
        return out

    LayerStack.forward = looped
    return orig, looped


def forward_selftest():
    """Confirm: (1) loop_k=1 patched == original output; (2) loop_k=2 runs + differs."""
    import torch
    from taco.model.tabpfn_arch.model.transformer import LayerStack
    global LOOP_K

    torch.manual_seed(0)
    # tiny fake stack of 3 identity-ish linear layers operating on (B,S,F,E)
    class Lin(torch.nn.Module):
        def __init__(s): super().__init__(); s.l = torch.nn.Linear(8, 8)
        def forward(s, x, **kw): return s.l(x)
    ls = LayerStack(layer_creator=Lin, num_layers=3, min_num_layers_layer_dropout=3)
    x = torch.randn(2, 5, 4, 8)
    orig = LayerStack.forward
    base = orig(ls, x, half_layers=True)

    install_looped_forward()
    LOOP_K = 1
    out1 = LayerStack.forward(ls, x, half_layers=True)
    LOOP_K = 2
    out2 = LayerStack.forward(ls, x, half_layers=True)
    d1 = (out1 - base).abs().max().item()
    d2 = (out2 - base).abs().max().item()
    print(f"loop_k=1 vs baseline maxdiff={d1:.2e} (must be ~0)")
    print(f"loop_k=2 vs baseline maxdiff={d2:.2e} (must be >0, loop active)")
    assert d1 < 1e-6, "loop_k=1 must equal baseline"
    assert d2 > 1e-4, "loop_k=2 must differ"
    print("FORWARD SELFTEST OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("gen-eval")
    pe.add_argument("--out_dir", required=True)
    pe.add_argument("--n_datasets", type=int, default=20)
    pe.add_argument("--seq_len", type=int, default=600)
    sub.add_parser("forward-selftest")
    a = p.parse_args()
    if a.cmd == "gen-eval":
        gen_eval(a.out_dir, n_datasets=a.n_datasets, seq_len=a.seq_len)
    elif a.cmd == "forward-selftest":
        forward_selftest()
