#!/bin/bash
# Description: Submits multi-agent PPO training or utility runs through Slurm.
#SBATCH --job-name=head_st_mappo
#SBATCH --partition=gpu_48
#SBATCH --gpus=a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

# Override these at submit time, e.g.:
#   CHECKPOINT_EVERY_UPDATES=10 sbatch multi_train.sh
RUN_MODE="${RUN_MODE:-train}"
CHECKPOINT_EVERY_UPDATES="${CHECKPOINT_EVERY_UPDATES:-25}"
SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-10}"
SHOW_VARIANCE="${SHOW_VARIANCE:-1}"

CMD=(
  python3 main.py
  --run-mode "${RUN_MODE}"
  --checkpoint-every-updates "${CHECKPOINT_EVERY_UPDATES}"
  --smoothing-window "${SMOOTHING_WINDOW}"
)

if [[ "${SHOW_VARIANCE}" == "1" ]]; then
  CMD+=(--show-variance)
fi

# Forward any extra CLI args passed to this script directly to main.py.
# Example:
#   sbatch multi_train.sh --plot-csv-path training_results/foo/training_metrics.csv --run-mode plot
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Launching command: ${CMD[*]}"
srun --ntasks=1 --nodes=1 "${CMD[@]}"
