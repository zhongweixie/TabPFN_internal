#!/usr/bin/env bash
# Fixed-large-k looped pretraining: loop_k=4 and loop_k=6, 80K steps each.
# Same recipe/seed/data as c1/c2/c3 (batch32, 80K) so results are directly comparable.
# Question: at FIXED (non-curriculum) training, does a larger loop_k keep helping past
# c2's k=2, or saturate / degrade? (c2@k2 mean-AUC 0.814 was best of c1/c2/c3.)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # exp/looped_pretrain/
REPRO="$(cd "$HERE/../.." && pwd)"                          # reproduce/
PY=/home/zxiebk/miniconda3/envs/wcb_rl_training/bin/python
export PYTHONPATH=/home/zxiebk/workspace/train/PFN/TACO/src:$HERE
EVAL="$HERE/eval_bank"; [ -d "$EVAL" ] || EVAL="$REPRO/eval_bank"   # held-out synth eval
STEPS=${STEPS:-80000}
SEED=42

launch() {  # $1=loop_k $2=nlayers $3=gpu
  local k=$1 L=$2 gpu=$3
  local name="loopk${k}_${L}L_${STEPS}"
  local ckdir="$REPRO/ckpt/$name"
  mkdir -p "$ckdir"
  CUDA_VISIBLE_DEVICES=$gpu LOOP_K=$k REINJECT_ALPHA=0.1 nohup $PY -u "$HERE/looped_train.py" \
    --nlayers $L --max_steps $STEPS --batch_size 32 --micro_batch_size 16 \
    --min_seq_len 128 --max_seq_len 600 --min_features 5 --max_features 40 \
    --device cuda --prior_device cpu --num_workers 4 \
    --np_seed $SEED --torch_seed $SEED --seed 0 \
    --wandb_log True --wandb_project tabpfn-looped-pretrain --wandb_name "$name" \
    --eval_every 1000 --save_temp_every 10000 --save_perm_every 20000 \
    --checkpoint_dir "$ckdir" \
    --eval_data_dir "$EVAL" --prior_type mix_scm \
    > "$REPRO/$name.out" 2>&1 &
  echo "launched $name (loop_k=$k nlayers=$L gpu=$gpu) pid $! -> ckpt/$name"
}

# GPUs 0 and 1 (free); both 12-layer so loop_k is the only variable vs c1/c2
launch 4 12 1
launch 6 12 4
echo "two fixed-large-k runs launched. logs: $REPRO/loopk{4,6}_12L_${STEPS}.out"
