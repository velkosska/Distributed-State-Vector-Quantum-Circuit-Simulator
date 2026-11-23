#!/usr/bin/env bash
#SBATCH --job-name=qsim-omp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64       # course QOS cap; 32 physical cores
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# OpenMP WEAK scaling on one node: N qubits at 1 thread, +1 qubit each time the
# thread count doubles (constant amplitudes per thread). Submit from the repo root.
#
#   N=24 sbatch slurm/openmp_scaling.sbatch                  # 24q at T=1 ... 30q at T=64
#   THREADS="1 8 64" N=20 bash slurm/openmp_scaling.sbatch   # (only inside an allocation)
#
# Strong scaling (fixed 28 qubits, close vs spread, core pinning) is
# slurm/omp_scaling.sbatch; this script refuses MODE=strong so the two scripts
# never write the same experiment.
# Course QOS "student": at most 64 CPUs per user (32 physical cores with 2 hardware
# threads each) and no --exclusive, or the job waits forever (QOSMaxCpuPerUserLimit).
#
# Build: cmake -S . -B build-omp -DQSIM_OPENMP=ON && cmake --build build-omp -j
set -euo pipefail

MODE=${MODE:-weak}                   # weak only (strong: slurm/omp_scaling.sbatch)
N=${N:-24}                           # qubits at T=1
if [ "$MODE" != weak ]; then
  echo "openmp_scaling.sbatch runs weak scaling only; use slurm/omp_scaling.sbatch for strong scaling" >&2
  exit 2
fi
KINDS=${KINDS:-"qft random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
MAX_T=$(python3 "${SLURM_SUBMIT_DIR:-.}/slurm/lib/physical_cores.py" | tr ',' '\n' | wc -l)   # physical cores
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
