#!/usr/bin/env bash
# Variable-loop-k CURRICULUM training (COCONUT-style): loop_k 1->2->3->4->5->6 in
# equal-step stages over 80K steps, rebuilding optimizer+scheduler+scaler at each switch.
# Same recipe/seed/data as c1/c2/c3/k4/k6 so the resulting variable-k model is comparable.
# Purpose: a single model that works across all loop_k (prerequisite for a binary-classifier
# adaptive-depth head; fixed-k models degenerate off-k, so maxConf failed).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$(cd "$HERE/../.." && pwd)"
PY=/home/zxiebk/miniconda3/envs/wcb_rl_training/bin/python
export PYTHONPATH=/home/zxiebk/workspace/train/PFN/TACO/src:$HERE
EVAL="$HERE/eval_bank"; [ -d "$EVAL" ] || EVAL="$REPRO/eval_bank"
STEPS=${STEPS:-80000}
GPU=${GPU:-2}
SEED=42
NAME="curric_k1to6_12L_${STEPS}"
CKDIR="$REPRO/ckpt/$NAME"
mkdir -p "$CKDIR"

CUDA_VISIBLE_DEVICES=$GPU LOOPK_SCHED="1,2,3,4,5,6" REINJECT_ALPHA=0.1 \
  nohup $PY -u "$HERE/train_curriculum.py" \
  --nlayers 12 --max_steps $STEPS --batch_size 32 --micro_batch_size 16 \
  --min_seq_len 128 --max_seq_len 600 --min_features 5 --max_features 40 \
  --device cuda --prior_device cpu --num_workers 4 \
  --np_seed $SEED --torch_seed $SEED --seed 0 \
  --wandb_log True --wandb_project tabpfn-looped-pretrain --wandb_name "$NAME" \
  --eval_every 1000 --save_temp_every 10000 --save_perm_every 20000 \
  --checkpoint_dir "$CKDIR" \
  --eval_data_dir "$EVAL" --prior_type mix_scm \
  > "$REPRO/$NAME.out" 2>&1 &
echo "launched $NAME (gpu=$GPU, loop_k 1->6) pid $! -> ckpt/$NAME"
echo "log: $REPRO/$NAME.out"
