#!/usr/bin/env bash
#SBATCH --job-name=qsim-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2        # one physical core (both hardware threads), 1 OpenMP thread
#SBATCH --mem=16G                # 28 qubits = 4 GiB state
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Serial (1-core) baseline: QFT, Grover and random circuits at 20..28 qubits,
# REPEATS runs each, with per-gate timings. Submit from the repo root.
#
#   sbatch slurm/baseline.sbatch                         # defaults below
#   QUBITS="20 24" REPEATS=1 sbatch slurm/baseline.sbatch
#   QUBITS="10 12" bash slurm/baseline.sbatch            # quick local test, no Slurm
#
# Course QOS "student": at most 64 CPUs per user (32 physical cores with 2 hardware
# threads each) and no --exclusive, or the job waits forever (QOSMaxCpuPerUserLimit).
# Other users' jobs can share the node's memory bandwidth: see methodology §7.
#
# Output: results/baseline/<jobid>/{meta.txt,runs.csv,raw/}  (see slurm/lib/run_harness.sh)
set -euo pipefail

QUBITS=${QUBITS:-"20 22 24 26 28"}
KINDS=${KINDS:-"qft grover random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
GROVER_ITERS=${GROVER_ITERS:-1}   # a full Grover search at 28q would take hours; 1 iteration is enough for time/gate

HARNESS_SCRIPT=slurm/baseline.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/lib/run_harness.sh"
export OMP_NUM_THREADS=1
BACKEND=${BACKEND:-serial}

max_q=$(tr ' ' '\n' <<<"$QUBITS" | sort -n | tail -1)
harness_guard_local "$max_q"
harness_init baseline

for kind in $KINDS; do
  for n in $QUBITS; do
    case $kind in
      qft)    circ=$(harness_gen_circuit qft "$n" --seed "$SEED"); extra=() ;;
      grover) circ=$(harness_gen_circuit grover "$n" --seed "$SEED" --iterations "$GROVER_ITERS"); extra=(iterations="$GROVER_ITERS") ;;
      random) circ=$(harness_gen_circuit random "$n" --seed "$SEED" --depth "$RANDOM_DEPTH"); extra=(depth="$RANDOM_DEPTH") ;;
      *) echo "unknown kind $kind" >&2; exit 1 ;;
    esac
    for r in $(seq 1 "$REPEATS"); do
      harness_run kind="$kind" qubits="$n" repeat="$r" seed="$SEED" threads=1 circuit="$circ" "${extra[@]}"
    done
  done
done

harness_finish
