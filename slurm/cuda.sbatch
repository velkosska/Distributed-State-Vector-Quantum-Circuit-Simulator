#!/usr/bin/env bash
#SBATCH --job-name=qsim-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Single-GPU runs: time per gate vs qubits (state stays on the device).
# Submit from the repo root:  QUBITS="20 24 28 30" sbatch slurm/cuda.sbatch
#
# Expects Emma's build at build-cuda/qsim_cuda with the same CLI and SUMMARY
# line. If it reports host<->device copy time, print it as extra SUMMARY keys
# (e.g. h2d_s= d2h_s=); the harness keeps the full stdout in raw/.
set -euo pipefail

QUBITS=${QUBITS:-"20 22 24 26 28 30"}
KINDS=${KINDS:-"qft grover random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
GROVER_ITERS=${GROVER_ITERS:-1}
export QSIM_BIN=${QSIM_BIN:-build-cuda/qsim_cuda}
export BACKEND=cuda
export QSIM_MODULES_EXTRA=${QSIM_MODULES_EXTRA:-cuda}
export OMP_NUM_THREADS=1

HARNESS_SCRIPT=slurm/cuda.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/lib/run_harness.sh"
max_q=$(tr ' ' '\n' <<<"$QUBITS" | sort -n | tail -1)
harness_guard_local "$max_q"
harness_init cuda

for kind in $KINDS; do
  for n in $QUBITS; do
    case $kind in
      qft)    circ=$(harness_gen_circuit qft "$n" --seed "$SEED"); extra=() ;;
      grover) circ=$(harness_gen_circuit grover "$n" --seed "$SEED" --iterations "$GROVER_ITERS"); extra=(iterations="$GROVER_ITERS") ;;
      random) circ=$(harness_gen_circuit random "$n" --seed "$SEED" --depth "$RANDOM_DEPTH"); extra=(depth="$RANDOM_DEPTH") ;;
    esac
    for r in $(seq 1 "$REPEATS"); do
      harness_run kind="$kind" qubits="$n" repeat="$r" seed="$SEED" gpus=1 circuit="$circ" "${extra[@]}"
    done
  done
done

harness_finish
