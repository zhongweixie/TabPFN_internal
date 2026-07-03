#!/usr/bin/env bash
# Launch ALL halting-mechanism experiments (H1/H2/H3/H5/H6) in parallel across GPUs.
# H1+H2: zero-retrain eval on curric-60K  (GPU 5, ~2-4h)
# H3:    halt MLP fine-tune from curric-60K (GPU 4, ~3h)
# H5-l001: ACT lambda=0.01 full retrain   (GPU 0, ~7 days)
# H5-l005: ACT lambda=0.05 full retrain   (GPU 1, ~7 days)
# H5-l01:  ACT lambda=0.1  full retrain   (GPU 2, ~7 days)
# H5-l02:  ACT lambda=0.2  full retrain   (GPU 3, ~7 days)
# H6:    Budget-conditional full retrain   (GPU 6, ~7 days)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$(cd "$HERE/../.." && pwd)"
PY=/home/zxiebk/miniconda3/envs/wcb_rl_training/bin/python
export PYTHONPATH=/home/zxiebk/workspace/train/PFN/TACO/src:$HERE
EVAL="$HERE/eval_bank"; [ -d "$EVAL" ] || EVAL="$REPRO/eval_bank"
STEPS=${STEPS:-80000}
SEED=42
CURRIC_60K="$REPRO/ckpt/curric_k1to6_12L_80000/step-60000.ckpt"

COMMON_TRAIN_ARGS="--nlayers 12 --max_steps $STEPS --batch_size 32 --micro_batch_size 16 \
  --min_seq_len 128 --max_seq_len 600 --min_features 5 --max_features 40 \
  --device cuda --prior_device cpu --num_workers 4 \
  --np_seed $SEED --torch_seed $SEED --seed 0 \
  --wandb_log True --wandb_project tabpfn-halting \
  --eval_every 1000 --save_temp_every 10000 --save_perm_every 20000 \
  --eval_data_dir $EVAL --prior_type mix_scm"

# ── H1/H2: zero-retrain eval ───────────────────────────────────────────────
echo "=== Launching H1+H2 eval (GPU 5) ==="
CUDA_VISIBLE_DEVICES=5 \
  nohup $PY -u "$HERE/run_h1_h2_eval.py" \
    --ckpt "$CURRIC_60K" \
    --methods h1,h2 \
    --out_json "$REPRO/h1h2_results.json" \
  > "$REPRO/h1h2_eval.out" 2>&1 &
echo "  pid=$! log=$REPRO/h1h2_eval.out"

# ── H3: halt MLP fine-tune ─────────────────────────────────────────────────
NAME_H3="h3_halt_mlp_10k"
CKDIR_H3="$REPRO/ckpt/$NAME_H3"
mkdir -p "$CKDIR_H3"
echo "=== Launching H3 halt MLP fine-tune (GPU 4) ==="
CUDA_VISIBLE_DEVICES=4 \
  HALT_LAMBDA=0.3 UNFREEZE_LAYERS=4 HALT_STEPS=10000 \
  HALT_K_SCHEDULE="1,2,3,4,5,6" \
  nohup $PY -u "$HERE/train_h3_halt_mlp.py" \
    --base_ckpt "$CURRIC_60K" \
    --halt_lambda 0.3 \
    --unfreeze_layers 4 \
    --halt_steps 10000 \
    $COMMON_TRAIN_ARGS \
    --wandb_name "$NAME_H3" \
    --checkpoint_dir "$CKDIR_H3" \
  > "$REPRO/${NAME_H3}.out" 2>&1 &
echo "  pid=$! log=$REPRO/${NAME_H3}.out"

# ── H5: ACT lambda search (4 variants) ────────────────────────────────────
for LAMBDA_STR in "001:0.01:0" "005:0.05:1" "01:0.1:2" "02:0.2:3"; do
  LS=$(echo $LAMBDA_STR | cut -d: -f1)
  LV=$(echo $LAMBDA_STR | cut -d: -f2)
  GPU=$(echo $LAMBDA_STR | cut -d: -f3)
  NAME_H5="h5_act_l${LS}_12L_${STEPS}"
  CKDIR_H5="$REPRO/ckpt/$NAME_H5"
  mkdir -p "$CKDIR_H5"
  echo "=== Launching H5 ACT lambda=${LV} (GPU ${GPU}) ==="
  CUDA_VISIBLE_DEVICES=$GPU \
    ACT_LAMBDA=$LV ACT_K_MAX=6 REINJECT_ALPHA=0.1 \
    nohup $PY -u "$HERE/train_h5_act.py" \
      $COMMON_TRAIN_ARGS \
      --wandb_name "$NAME_H5" \
      --checkpoint_dir "$CKDIR_H5" \
    > "$REPRO/${NAME_H5}.out" 2>&1 &
  echo "  pid=$! log=$REPRO/${NAME_H5}.out"
done

# ── H6: Budget-conditional full retrain ────────────────────────────────────
NAME_H6="h6_budget_cond_12L_${STEPS}"
CKDIR_H6="$REPRO/ckpt/$NAME_H6"
mkdir -p "$CKDIR_H6"
echo "=== Launching H6 budget-conditional (GPU 6) ==="
CUDA_VISIBLE_DEVICES=6 \
  BUDGET_K_MAX=6 BUDGET_SAMPLE_MODE=uniform REINJECT_ALPHA=0.1 \
  nohup $PY -u "$HERE/train_h6_budget_cond.py" \
    $COMMON_TRAIN_ARGS \
    --wandb_name "$NAME_H6" \
    --checkpoint_dir "$CKDIR_H6" \
  > "$REPRO/${NAME_H6}.out" 2>&1 &
echo "  pid=$! log=$REPRO/${NAME_H6}.out"

echo ""
echo "=== All experiments launched. Monitor with: ==="
echo "  tail -f $REPRO/h1h2_eval.out"
echo "  tail -f $REPRO/h3_halt_mlp_10k.out"
echo "  tail -f $REPRO/h5_act_l001_12L_${STEPS}.out"
echo "  tail -f $REPRO/h6_budget_cond_12L_${STEPS}.out"
