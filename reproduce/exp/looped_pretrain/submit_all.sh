#!/usr/bin/env bash
# Submit all H5/H6 Slurm jobs for the 8xH800 cluster.
# Usage: bash submit_all.sh [--dry-run]
#
# GPU allocation: H5-l001(2) + H5-l005(2) + H6(4) = 8 GPUs total
# Expected runtime: ~24-36h on H800 (vs ~7 days on L20 single GPU)
#
# Before running:
#   export TACO_SRC=/path/to/TACO/src
#   export REPRO=/path/to/reproduce/dir
#   export CONDA_ENV=your_env_name

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=${1:-}

submit() {
  local script=$1
  if [[ "$DRY" == "--dry-run" ]]; then
    echo "[dry-run] sbatch $script"
  else
    sbatch "$HERE/$script"
    echo "Submitted: $script"
  fi
}

submit slurm_h5_l001.sh
submit slurm_h5_l005.sh
submit slurm_h6.sh

echo ""
echo "Monitor jobs: squeue -u $USER"
echo "Logs: <job-name>_<jobid>.out in current dir"
