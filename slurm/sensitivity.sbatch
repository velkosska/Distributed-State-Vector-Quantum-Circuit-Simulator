#!/usr/bin/env bash
#SBATCH --job-name=qsim-sens
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8      # P = nodes * ntasks-per-node ranks (power of two)
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Communication sensitivity of the MPI build. Fixed qubits N and ranks P = 2^p;
# what varies is the circuit:
#   * the three benchmark kinds (qft, grover, random), and
#   * random circuits whose share of gates on the p global qubits is set with
#     gen_circuits.py random --global-fraction F --global-qubits p.
# For each circuit, analyze_circuit.py --ranks P reports how many gates really
# need an exchange (pct_comm); that table is stored next to runs.csv as
# circuit_analysis.csv and joined by the analysis scripts (analysis/sensitivity.py).
#
#   sbatch slurm/sensitivity.sbatch                           # N=30, P=16
#   N=32 FRACTIONS="0 0.1 0.5 1" sbatch --nodes=4 slurm/sensitivity.sbatch
#
# Turn MPI qubit remapping OFF for this experiment (QSIM_ARGS=--no-remap),
# otherwise the binary moves gates off global qubits and pct_comm from the static
# analysis no longer matches what ran. Run it again with remapping ON to show the gain.
set -euo pipefail

N=${N:-30}
KINDS=${KINDS:-"qft grover random"}
FRACTIONS=${FRACTIONS:-"0 0.05 0.1 0.2 0.3 0.5 0.75 1"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
PPN=${SLURM_NTASKS_PER_NODE:-1}
NNODES=${SLURM_JOB_NUM_NODES:-1}
P=${P:-$((PPN * NNODES))}
SRUN_EXTRA=${SRUN_EXTRA:-"--cpu-bind=cores"}
export QSIM_BIN=${QSIM_BIN:-build-mpi/qsim_mpi}
export BACKEND=mpi
export QSIM_MODULES_EXTRA=${QSIM_MODULES_EXTRA:-mpi}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

HARNESS_SCRIPT=slurm/sensitivity.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/lib/run_harness.sh"

GP=0; x=$P; while [ "$x" -gt 1 ]; do x=$((x / 2)); GP=$((GP + 1)); done
[ $((1 << GP)) -eq "$P" ] || { echo "P=$P is not a power of two" >&2; exit 1; }
[ "$GP" -ge 1 ] || { echo "need P >= 2 ranks for a global qubit" >&2; exit 1; }
nodes=$(( (P + PPN - 1) / PPN ))
launch="srun --nodes=$nodes --ntasks=$P --cpus-per-task=$OMP_NUM_THREADS --exact $SRUN_EXTRA"
[ -n "${SLURM_JOB_ID:-}" ] || launch="mpirun -np $P"

harness_guard_local "$N"
harness_init sensitivity
echo "ranks=$P global_qubits=$GP" >> "$HARNESS_OUT/meta.txt"
ANALYSIS="$HARNESS_OUT/circuit_analysis.csv"

analyze() {   # append the analyze_circuit.py row for (circuit, P), with the circuit path as key
  local circ=$1 tmp
  tmp=$(mktemp)
  python3 src/tools/analyze_circuit.py "$circ" --ranks "$P" --csv "$tmp" > /dev/null
  if [ ! -s "$ANALYSIS" ]; then sed -n '1s/^/circuit_path,/p' "$tmp" > "$ANALYSIS"; fi
  tail -n +2 "$tmp" | sed "s|^|$circ,|" >> "$ANALYSIS"
  rm -f "$tmp"
}

run_all() {   # run_all CIRCUIT key=value...
  local circ=$1; shift
  analyze "$circ"
  for r in $(seq 1 "$REPEATS"); do
    harness_run qubits="$N" repeat="$r" seed="$SEED" ranks="$P" nodes="$nodes" threads="$OMP_NUM_THREADS" \
                global_qubits="$GP" circuit="$circ" launch="$launch" "$@"
  done
}

# 1) the benchmark circuits as they are
for kind in $KINDS; do
  case $kind in
    qft)    run_all "$(harness_gen_circuit qft "$N" --seed "$SEED")" kind=qft tag=kind ;;
    grover) run_all "$(harness_gen_circuit grover "$N" --seed "$SEED" --iterations 1)" kind=grover iterations=1 tag=kind ;;
    random) run_all "$(harness_gen_circuit random "$N" --seed "$SEED" --depth "$RANDOM_DEPTH")" kind=random depth="$RANDOM_DEPTH" tag=kind ;;
  esac
done

# 2) random circuits with a controlled share of gates on global qubits
for f in $FRACTIONS; do
  circ=$(harness_gen_circuit random "$N" --seed "$SEED" --depth "$RANDOM_DEPTH" --global-fraction "$f" --global-qubits "$GP")
  run_all "$circ" kind=random_gf depth="$RANDOM_DEPTH" global_fraction="$f" tag=global_fraction
done

harness_finish
