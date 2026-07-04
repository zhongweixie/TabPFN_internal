#!/usr/bin/env bash
#SBATCH --job-name=h5_act_l005
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

TACO_SRC="${TACO_SRC:-/home/$USER/workspace/train/PFN/TACO/src}"
CONDA_ENV="${CONDA_ENV:-wcb_rl_training}"
REPRO="${REPRO:-/home/$USER/workspace/train/PFN/TabPFN/reproduce}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$TACO_SRC:$HERE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

BASE_CKPT="$REPRO/ckpt/curric_k1to6_12L_80000/step-60000.ckpt"
NAME="h5_act_l005_12L_80000"
CKDIR="$REPRO/ckpt/$NAME"
mkdir -p "$CKDIR"

export ACT_LAMBDA=0.05 ACT_K_MAX=6 REINJECT_ALPHA=0.1 ACT_BASE_CKPT="$BASE_CKPT"

torchrun \
  --standalone --nnodes=1 --nproc_per_node=2 \
  "$HERE/train_h5_act.py" \
  --nlayers 12 --max_steps 80000 --batch_size 32 --micro_batch_size 16 \
  --min_seq_len 128 --max_seq_len 600 --min_features 5 --max_features 40 \
  --device cuda --prior_device cpu --num_workers 4 \
  --np_seed 42 --torch_seed 42 --seed 0 \
  --wandb_log True --wandb_project tabpfn-halting --wandb_name "$NAME" \
  --eval_every 1000 --save_temp_every 5000 --save_perm_every 10000 \
  --eval_data_dir "$REPRO/eval_bank" --prior_type mix_scm \
  --checkpoint_dir "$CKDIR"

echo "[$NAME] DONE"
