#!/bin/bash
# Description: Slurm job for NCSA Delta to train / simulate the follower-UAV MAPPO policy.
#
# Usage (from the project directory on a Delta login node):
#   sbatch --account=<alloc>-gpu ncsa_train.sh
#   RUN_MODE=s EVAL_CKPT=training_results/<run>/checkpoint.pt sbatch --account=<alloc>-gpu ncsa_train.sh
#   sbatch --account=<alloc>-gpu ncsa_train.sh -nu 6 -ns 50 -hm c   # extra args go to main.py
# Find your allocation account name with: accounts
#
# CPU-only alternative: sbatch --partition=cpu --account=<alloc>-cpu --gpus-per-node=0 ncsa_train.sh
#SBATCH --job-name=mappo_followers
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Slurm opens the log files before the script runs, so create slurm_logs/ before submitting:
#   mkdir -p slurm_logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

# Python environment (created once on a login node, see the setup notes below).
ENV_DIR="${ENV_DIR:-$HOME/envs/mappo}"
module purge
module load python/3.11 2>/dev/null || module load python 2>/dev/null || true
if [[ ! -d "${ENV_DIR}" ]]; then
  echo "Missing virtualenv at ${ENV_DIR}. One-time setup on a login node:" >&2
  echo "  python3 -m venv ${ENV_DIR} && source ${ENV_DIR}/bin/activate" >&2
  echo "  pip install torch numpy scipy matplotlib pyro-ppl pandas pillow" >&2
  exit 1
fi
source "${ENV_DIR}/bin/activate"

# Keep BLAS/OpenMP thread counts matched to the allocation; run headless.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

RUN_MODE="${RUN_MODE:-t}"          # t=train, s=sim, e=eval, p=plot
EVAL_CKPT="${EVAL_CKPT:-}"         # checkpoint for sim/eval modes

CMD=(python3 main.py -r "${RUN_MODE}")
if [[ -n "${EVAL_CKPT}" ]]; then
  CMD+=(-ec "${EVAL_CKPT}")
fi
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Command: ${CMD[*]}"
nvidia-smi 2>/dev/null || echo "No GPU visible"

srun "${CMD[@]}"
