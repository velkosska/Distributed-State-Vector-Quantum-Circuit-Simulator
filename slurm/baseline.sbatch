#!/usr/bin/env bash
#SBATCH --job-name=qsim-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive              # whole node: no neighbours competing for memory bandwidth
#SBATCH --mem=16G                # 28 qubits = 4 GiB state
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Serial (1-core) baseline: QFT, Grover and random circuits at 20..28 qubits,
# REPEATS runs each, with per-gate timings.
#
#   sbatch slurm/baseline.sbatch                         # defaults below
#   QUBITS="20 24" REPEATS=1 sbatch slurm/baseline.sbatch
#   QUBITS="10 12" bash slurm/baseline.sbatch            # quick local test, no Slurm
#
# Build first (on the login node is fine, it is light):
#   source env/modules.sh && cmake -S . -B build && cmake --build build -j

set -euo pipefail

QUBITS=${QUBITS:-"20 22 24 26 28"}
KINDS=${KINDS:-"qft grover random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
GROVER_ITERS=${GROVER_ITERS:-1}   # a full Grover search at 28q would take hours; 1 iteration is enough for time/gate
QSIM_BIN=${QSIM_BIN:-build/qsim}

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
[ -f env/modules.sh ] && command -v module >/dev/null 2>&1 && source env/modules.sh

export OMP_NUM_THREADS=1
JOBID=${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}
OUT="results/baseline/$JOBID"
mkdir -p "$OUT/raw" data/baseline

# ---------------------------------------------------------------- metadata
{
  echo "job_id:       $JOBID"
  echo "date:         $(date -Iseconds)"
  echo "host:         $(hostname)"
  echo "cpu:          $(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p')"
  echo "git_commit:   $(git rev-parse --verify -q HEAD || echo unknown)"
  echo "git_dirty:    $(git status --porcelain 2>/dev/null | grep -q . && echo yes || echo no)"
  echo "binary:       $QSIM_BIN"
  echo "qubits:       $QUBITS"
  echo "kinds:        $KINDS"
  echo "repeats:      $REPEATS"
  echo "seed:         $SEED"
  echo "random_depth: $RANDOM_DEPTH"
  echo "grover_iters: $GROVER_ITERS"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "--- module list"
  if command -v module >/dev/null 2>&1; then module list 2>&1; else echo "module command not available"; fi
  echo "--- commands"
} > "$OUT/meta.txt"

[ -x "$QSIM_BIN" ] || { echo "missing $QSIM_BIN, build first" >&2; exit 1; }

echo "kind,qubits,repeat,seed,gates,total_s,mean_gate_s,norm_err,circuit" > "$OUT/runs.csv"

for kind in $KINDS; do
  for n in $QUBITS; do
    circ="data/baseline/${kind}_${n}_s${SEED}.txt"
    case $kind in
      qft)    gen=(qft -n "$n" --seed "$SEED") ;;
      grover) gen=(grover -n "$n" --seed "$SEED" --iterations "$GROVER_ITERS") ;;
      random) gen=(random -n "$n" --seed "$SEED" --depth "$RANDOM_DEPTH") ;;
      *) echo "unknown kind $kind" >&2; exit 1 ;;
    esac
    python3 src/tools/gen_circuits.py "${gen[@]}" -o "$circ"

    for r in $(seq 1 "$REPEATS"); do
      cmd=("$QSIM_BIN" "$circ" --timing "$OUT/raw/${kind}_${n}_r${r}.csv")
      echo "${cmd[*]}" >> "$OUT/meta.txt"
      line=$("${cmd[@]}" | tee -a "$OUT/raw/stdout.txt" | grep '^SUMMARY')
      get() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<<"$line"; }
      echo "$kind,$n,$r,$SEED,$(get gates_run),$(get total_s),$(get mean_gate_s),$(get norm_err),$circ" >> "$OUT/runs.csv"
      echo "[$(date +%T)] $kind n=$n rep=$r total=$(get total_s)s"
    done
  done
done

echo "done: $OUT"
