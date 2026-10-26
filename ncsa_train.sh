#!/bin/bash
# Submit MAPPO training or utility runs to the NCSA Illinois Campus Cluster:
#   sbatch ncsa_train.sh
# Forward additional main.py arguments after the script name.
#   sbatch ncsa_train.sh --epochs 100 --n-uavs 6 --n-sensors 50
# Use RUN_MODE=t|s|e|p for train|sim|eval|plot.
# The trainer currently uses CPU, so this job does not reserve a GPU.
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=uav-mappo
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project directory}"

PYTHON_BIN="${PYTHON_BIN:-$SLURM_SUBMIT_DIR/../.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to the project's Python environment when submitting." >&2
    exit 1
fi

RUN_MODE="${RUN_MODE:-t}"
case "$RUN_MODE" in
    train) RUN_MODE=t ;;
    sim)   RUN_MODE=s ;;
    eval)  RUN_MODE=e ;;
    plot)  RUN_MODE=p ;;
    t|s|e|p) ;;
    *)
        echo "ERROR: RUN_MODE must be train/t, sim/s, eval/e, or plot/p." >&2
        exit 2
        ;;
esac

CHECKPOINT_EVERY_UPDATES="${CHECKPOINT_EVERY_UPDATES:-25}"
SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-10}"
SHOW_VARIANCE="${SHOW_VARIANCE:-1}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CMD=(
    "$PYTHON_BIN" main.py
    --run-mode "$RUN_MODE"
    --checkpoint-every-updates "$CHECKPOINT_EVERY_UPDATES"
    --smoothing-window "$SMOOTHING_WINDOW"
)

if [[ "$SHOW_VARIANCE" == "1" ]]; then
    CMD+=(--show-variance)
fi

CMD+=("$@")

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_JOB_NODELIST:-unknown}"
echo "Python: $PYTHON_BIN"
echo "Launching: ${CMD[*]}"
srun --nodes=1 --ntasks=1 "${CMD[@]}"
