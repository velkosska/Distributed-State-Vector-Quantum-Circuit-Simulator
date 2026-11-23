#!/usr/bin/env bash
#SBATCH --job-name=qsim-omp-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#
# OpenMP strong scaling, 28 qubits, close vs spread. Submit from the repo root.
# Course QOS "student": at most 64 CPUs per user (32 physical cores with 2 hardware
# threads each) and no --exclusive, or the job waits forever (QOSMaxCpuPerUserLimit).
# 64 CPUs = 32 physical cores, so the default sweep stops at 32 threads.
set -euo pipefail

N=${N:-28}
THREADS=${THREADS:-"1 2 4 8 16 32"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
export OMP_DYNAMIC=false OMP_PLACES=cores OMP_DISPLAY_AFFINITY=true
unset OMP_THREAD_LIMIT GOMP_CPU_AFFINITY

HARNESS_SCRIPT=slurm/omp_scaling.sbatch
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/slurm/lib/run_harness.sh"
harness_guard_local "$N"
cd "$ROOT"
harness_load_modules
cores=$(python3 slurm/lib/physical_cores.py)
IFS=, read -r -a places <<<"$cores"
for t in $THREADS; do
  if (( t > ${#places[@]} )); then
    _h_die "$t threads requested, but only ${#places[@]} physical cores are available"
  fi
done

build="build-omp-${SLURM_JOB_ID:-local}"
configure=(cmake -S . -B "$build" -DQSIM_OPENMP=ON)
compile=(cmake --build "$build" --parallel 4 --target qsim)
"${configure[@]}"
"${compile[@]}"
export QSIM_BIN="$build/qsim" BACKEND=openmp LAUNCHER="taskset --cpu-list $cores"
export OMP_PROC_BIND=close
harness_init openmp_strong
_h_meta "configure: ${configure[*]}"
_h_meta "build:     ${compile[*]}"
_h_meta "settings:  qubits=$N threads=($THREADS) repeats=$REPEATS seed=$SEED"

qft=$(harness_gen_circuit qft "$N" --seed "$SEED")
random=$(harness_gen_circuit random "$N" --seed "$SEED" --depth "$RANDOM_DEPTH")
for t in $THREADS; do
  export OMP_NUM_THREADS=$t
  for bind in close spread; do
    export OMP_PROC_BIND=$bind
    for r in $(seq 1 "$REPEATS"); do
      harness_run kind=qft qubits="$N" repeat="$r" seed="$SEED" circuit="$qft" tag="$bind"
      harness_run kind=random qubits="$N" repeat="$r" seed="$SEED" circuit="$random" tag="$bind" depth="$RANDOM_DEPTH"
    done
  done
done

harness_finish
python3 analysis/omp_scaling.py "$HARNESS_OUT" --repeats "$REPEATS"
