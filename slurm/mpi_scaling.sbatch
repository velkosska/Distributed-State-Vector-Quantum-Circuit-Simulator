#!/usr/bin/env bash
#SBATCH --job-name=qsim-mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16     # ranks per node (power of two); threads per rank below
#SBATCH --cpus-per-task=2        # one physical core per rank (OpenMP threads per rank = this)
#SBATCH --mem=100G               # per node; 33 qubits on 32 ranks = 64 GiB per node
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# MPI strong and weak scaling. One srun step per rank count, ranks P = 1,2,4,..
# up to the allocation. Submit from the repo root.
#
#   MODE=strong N=30 sbatch slurm/mpi_scaling.sbatch          # fixed 30 qubits, P = 1..32 (16 per node)
#   MODE=weak   N=28 sbatch slurm/mpi_scaling.sbatch          # 28q at P=1, +1 qubit per doubling
#   MODE=strong N=30 sbatch --ntasks-per-node=8 slurm/mpi_scaling.sbatch   # P=16 spread over 2 nodes
#   MODE=strong N=30 sbatch --nodes=1 --ntasks-per-node=32 slurm/mpi_scaling.sbatch  # 1 node only
#
# Course QOS "student": at most 64 CPUs per user (32 physical cores with 2 hardware
# threads each) and no --exclusive, or the job waits forever (QOSMaxCpuPerUserLimit).
# nodes x ntasks-per-node x cpus-per-task must stay <= 64.
#
# Expects Edard's MPI build at build-mpi/qsim_mpi with the same CLI and
# SUMMARY line as build/qsim (add ranks=/threads= to SUMMARY if you can).
set -euo pipefail

MODE=${MODE:-strong}
N=${N:-30}
KINDS=${KINDS:-"qft random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
PPN=${SLURM_NTASKS_PER_NODE:-1}                 # ranks per node
NNODES=${SLURM_JOB_NUM_NODES:-1}
MAX_P=$((PPN * NNODES))
RANKS=${RANKS:-}
SRUN_EXTRA=${SRUN_EXTRA:-"--cpu-bind=cores"}
export QSIM_BIN=${QSIM_BIN:-build-mpi/qsim_mpi}
export BACKEND=mpi
export QSIM_MODULES_EXTRA=${QSIM_MODULES_EXTRA:-mpi}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

HARNESS_SCRIPT=slurm/mpi_scaling.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/lib/run_harness.sh"
[ -n "$RANKS" ] || RANKS=$(harness_pow2_upto "$MAX_P")

log2() { local x=$1 l=0; while [ "$x" -gt 1 ]; do x=$((x / 2)); l=$((l + 1)); done; echo "$l"; }
qubits_for() { if [ "$MODE" = weak ]; then echo $((N + $(log2 "$1"))); else echo "$N"; fi; }
max_p=$(tr ' ' '\n' <<<"$RANKS" | sort -n | tail -1)
harness_guard_local "$(qubits_for "$max_p")"
harness_init "mpi_${MODE}"

for kind in $KINDS; do
  for p in $RANKS; do
    n=$(qubits_for "$p")
    nodes=$(( (p + PPN - 1) / PPN ))
    case $kind in
      qft)    circ=$(harness_gen_circuit qft "$n" --seed "$SEED"); extra=() ;;
      grover) circ=$(harness_gen_circuit grover "$n" --seed "$SEED" --iterations 1); extra=(iterations=1) ;;
      random) circ=$(harness_gen_circuit random "$n" --seed "$SEED" --depth "$RANDOM_DEPTH"); extra=(depth="$RANDOM_DEPTH") ;;
    esac
    launch="srun --nodes=$nodes --ntasks=$p --cpus-per-task=$OMP_NUM_THREADS --exact $SRUN_EXTRA"
    [ -n "${SLURM_JOB_ID:-}" ] || launch="mpirun -np $p"     # local test inside an interactive shell
    for r in $(seq 1 "$REPEATS"); do
      harness_run kind="$kind" qubits="$n" repeat="$r" seed="$SEED" ranks="$p" nodes="$nodes" \
                  threads="$OMP_NUM_THREADS" circuit="$circ" tag="$MODE" launch="$launch" "${extra[@]}"
    done
  done
done

harness_finish
