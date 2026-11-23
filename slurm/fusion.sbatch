#!/usr/bin/env bash
#SBATCH --job-name=qsim-fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64       # course QOS cap: 64 CPUs = 32 physical cores
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Optimisation 1, gate fusion: every circuit is run with --fusion none and
# --fusion 1q, on 1 thread (serial reference) and on all cores of the node.
# Same circuits and seeds as slurm/baseline.sbatch. Submit from the repo root.
# Course QOS "student": at most 64 CPUs per user (32 physical cores with 2 hardware
# threads each) and no --exclusive, or the job waits forever (QOSMaxCpuPerUserLimit).
#
#   sbatch slurm/fusion.sbatch
#   QUBITS="24 26" THREADS="1" sbatch slurm/fusion.sbatch
#   QUBITS="10 12" THREADS="1 2" REPEATS=1 bash slurm/fusion.sbatch   # quick local test
#
# Output: results/fusion/<jobid>/{meta.txt,runs.csv,raw/}; the `fusion`,
# `gates_in` and `gates_run` columns tell the two modes apart.
# Summarise with: python3 analysis/fusion.py results/fusion/<jobid>
set -euo pipefail

QUBITS=${QUBITS:-"24 26 28"}
KINDS=${KINDS:-"qft grover random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
GROVER_ITERS=${GROVER_ITERS:-1}
MAX_T=$(python3 "${SLURM_SUBMIT_DIR:-.}/slurm/lib/physical_cores.py" | tr ',' '\n' | wc -l)   # physical cores
THREADS=${THREADS:-"1 $MAX_T"}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores} OMP_DYNAMIC=false

HARNESS_SCRIPT=slurm/fusion.sbatch
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/slurm/lib/run_harness.sh"
max_q=$(tr ' ' '\n' <<<"$QUBITS" | sort -n | tail -1)
harness_guard_local "$max_q"
cd "$ROOT"
harness_load_modules

# Fresh OpenMP build of this commit for this job (1 thread == serial code path).
build="build-fusion-${SLURM_JOB_ID:-local}"
configure=(cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DQSIM_OPENMP=ON)
compile=(cmake --build "$build" --parallel 4 --target qsim)
"${configure[@]}" > /dev/null
"${compile[@]}" > /dev/null
export QSIM_BIN="$build/qsim" BACKEND=openmp GATE_TIMING=${GATE_TIMING:-0}

harness_init fusion
_h_meta "configure: ${configure[*]}"
_h_meta "build:     ${compile[*]}"
_h_meta "settings:  qubits=($QUBITS) kinds=($KINDS) threads=($THREADS) repeats=$REPEATS seed=$SEED"

for kind in $KINDS; do
  for n in $QUBITS; do
    case $kind in
      qft)    circ=$(harness_gen_circuit qft "$n" --seed "$SEED"); extra=() ;;
      grover) circ=$(harness_gen_circuit grover "$n" --seed "$SEED" --iterations "$GROVER_ITERS"); extra=(iterations="$GROVER_ITERS") ;;
      random) circ=$(harness_gen_circuit random "$n" --seed "$SEED" --depth "$RANDOM_DEPTH"); extra=(depth="$RANDOM_DEPTH") ;;
      *) echo "unknown kind $kind" >&2; exit 1 ;;
    esac
    for t in $THREADS; do
      export OMP_NUM_THREADS=$t
      for r in $(seq 1 "$REPEATS"); do
        for mode in none 1q; do   # interleaved, so drift on the node hits both modes alike
          harness_run kind="$kind" qubits="$n" repeat="$r" seed="$SEED" threads="$t" circuit="$circ" \
                      tag="fusion_$mode" "${extra[@]}" -- --fusion "$mode"
        done
      done
    done
  done
done

harness_finish
python3 analysis/fusion.py "$HARNESS_OUT" || true
