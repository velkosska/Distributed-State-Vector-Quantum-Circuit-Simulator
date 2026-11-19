#!/usr/bin/env bash
#SBATCH --job-name=qsim-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Correctness check against Qiskit (2..20 qubits, tolerance 1e-10).
# Run after any change to the kernels:  sbatch slurm/validate.sbatch
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
[ -f env/modules.sh ] && command -v module >/dev/null 2>&1 && source env/modules.sh
JOBID=${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}
OUT="results/validation/$JOBID"
mkdir -p "$OUT"
git rev-parse --verify -q HEAD > "$OUT/git_commit.txt" || echo unknown > "$OUT/git_commit.txt"
python3 src/tools/validate_qiskit.py --qsim "${QSIM_BIN:-build/qsim}" \
  --max-qubits "${MAX_QUBITS:-20}" --tol 1e-10 --csv "$OUT/validation.csv" | tee "$OUT/validation.txt"
