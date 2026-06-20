#!/usr/bin/env bash
# Extended step-2: 80K steps with wandb logging.
# Resume-safe: TACO auto-resumes from last checkpoint in --checkpoint_dir.
#   c2 looped (loop_k=2@12L, re-injection)
#   c3 deep   (loop_k=1@24L, matched compute)
# c1 baseline already at 10K; extend it too for fair comparison at same steps.
set -u
REPO=/data/zxiebk/workspace/train/PFN/TabPFN/reproduce
PY=/home/zxiebk/miniconda3/envs/wcb_rl_training/bin/python
export PYTHONPATH=/home/zxiebk/workspace/train/PFN/TACO/src:$REPO
EVAL=$REPO/eval_bank
STEPS=${STEPS:-80000}
SEED=42
WANDB_PROJECT="tabpfn-looped-pretrain"

common() {  # $1=loop_k $2=nlayers $3=gpu $4=tag $5=wandb_name
  CUDA_VISIBLE_DEVICES=$3 LOOP_K=$1 REINJECT_ALPHA=0.1 nohup $PY -u $REPO/looped_train.py \
    --nlayers $2 --max_steps $STEPS --batch_size 32 --micro_batch_size 16 \
    --min_seq_len 128 --max_seq_len 600 --min_features 5 --max_features 40 \
    --device cuda --prior_device cpu --num_workers 4 \
    --np_seed $SEED --torch_seed $SEED --seed 0 \
    --wandb_log True --wandb_project $WANDB_PROJECT --wandb_name "$5" \
    --eval_every 1000 --save_temp_every 5000 --save_perm_every 20000 \
    --checkpoint_dir $REPO/ckpt_$4 \
    --eval_data_dir $EVAL --prior_type mix_scm \
    > $REPO/ext_$4.out 2>&1 &
  echo "launched $4 ($5, loop_k=$1 nlayers=$2 gpu=$3) pid $!"
}

mkdir -p $REPO/ckpt_c1_base $REPO/ckpt_c2_loop $REPO/ckpt_c3_deep

common 1 12 0 c1_base "c1-base-k1@12L"
common 2 12 1 c2_loop "c2-loop-k2@12L"
common 1 24 5 c3_deep "c3-deep-k1@24L"
echo "all launched. wandb project: $WANDB_PROJECT | tail ext_*.out"
