#!/usr/bin/env bash
# Step-2 go/no-go matrix: 3 configs, same seed (identical train stream), same eval bank.
#   c1 baseline      : loop_k=1 @12L  (12 layer-applications)
#   c2 looped        : loop_k=2 @12L  (24 layer-applications, re-injection, SAME params as c1)
#   c3 deep control  : loop_k=1 @24L  (24 layer-applications, MATCHED compute, 2x params)
# GO iff c2 >= c3 on held-out eval (recurrence beats equal-compute depth) AND c2 > c1.
set -u
REPO=/data/zxiebk/workspace/train/PFN/TabPFN/reproduce
PY=/home/zxiebk/miniconda3/envs/wcb_rl_training/bin/python
export PYTHONPATH=/home/zxiebk/workspace/train/PFN/TACO/src:$REPO
EVAL=$REPO/eval_bank
STEPS=${STEPS:-10000}
SEED=42

common() {  # $1=loop_k $2=nlayers $3=gpu $4=tag
  CUDA_VISIBLE_DEVICES=$3 LOOP_K=$1 REINJECT_ALPHA=0.1 nohup $PY -u $REPO/looped_train.py \
    --nlayers $2 --max_steps $STEPS --batch_size 32 --micro_batch_size 16 \
    --min_seq_len 128 --max_seq_len 600 --min_features 5 --max_features 40 \
    --device cuda --prior_device cpu --num_workers 4 \
    --np_seed $SEED --torch_seed $SEED --seed 0 \
    --wandb_log False --eval_every 1000 --checkpoint_dir /tmp/step2_$4 \
    --eval_data_dir $EVAL --prior_type mix_scm \
    > $REPO/step2_$4.out 2>&1 &
  echo "launched $4 (loop_k=$1 nlayers=$2 gpu=$3) pid $!"
}

common 1 12 0 c1_base
common 2 12 1 c2_loop
common 1 24 5 c3_deep
echo "all launched. tail step2_*.out"
