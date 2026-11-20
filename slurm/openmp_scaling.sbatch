#!/usr/bin/env bash
#SBATCH --job-name=qsim-omp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128      # set to the core count of one node of your partition
#SBATCH --exclusive
#SBATCH --mem=0                  # all memory of the node
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# OpenMP strong and weak scaling on one node. Submit from the repo root.
#
#   MODE=strong N=28 sbatch slurm/openmp_scaling.sbatch      # fixed 28 qubits, T = 1,2,4,..
#   MODE=weak   N=24 sbatch slurm/openmp_scaling.sbatch      # 24q at T=1, +1 qubit per doubling
#   THREADS="1 8 64" MODE=strong bash slurm/openmp_scaling.sbatch   # (only inside an allocation)
#
# Build: cmake -S . -B build-omp -DQSIM_OPENMP=ON && cmake --build build-omp -j
set -euo pipefail

MODE=${MODE:-strong}                 # strong | weak
N=${N:-28}                           # strong: qubits; weak: qubits at T=1
KINDS=${KINDS:-"qft random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
MAX_T=${SLURM_CPUS_PER_TASK:-$(nproc)}
THREADS=${THREADS:-}
export QSIM_BIN=${QSIM_BIN:-build-omp/qsim}
export BACKEND=openmp
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_DYNAMIC=false

HARNESS_SCRIPT=slurm/openmp_scaling.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/lib/run_harness.sh"
[ -n "$THREADS" ] || THREADS=$(harness_pow2_upto "$MAX_T")

qubits_for() {   # qubits for T threads
  if [ "$MODE" = weak ]; then
    local t=$1 n=$N; while [ "$t" -gt 1 ]; do t=$((t / 2)); n=$((n + 1)); done; echo "$n"
  else echo "$N"; fi
}
max_t=$(tr ' ' '\n' <<<"$THREADS" | sort -n | tail -1)
harness_guard_local "$(qubits_for "$max_t")"
harness_init "openmp_${MODE}"

for kind in $KINDS; do
  for t in $THREADS; do
    n=$(qubits_for "$t")
    case $kind in
      qft)    circ=$(harness_gen_circuit qft "$n" --seed "$SEED"); extra=() ;;
      grover) circ=$(harness_gen_circuit grover "$n" --seed "$SEED" --iterations 1); extra=(iterations=1) ;;
      random) circ=$(harness_gen_circuit random "$n" --seed "$SEED" --depth "$RANDOM_DEPTH"); extra=(depth="$RANDOM_DEPTH") ;;
    esac
    export OMP_NUM_THREADS=$t
    for r in $(seq 1 "$REPEATS"); do
      harness_run kind="$kind" qubits="$n" repeat="$r" seed="$SEED" threads="$t" circuit="$circ" \
                  tag="$MODE" "${extra[@]}"
    done
  done
done

harness_finish
