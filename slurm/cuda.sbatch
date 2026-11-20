#!/usr/bin/env bash
#SBATCH --job-name=qsim-cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000bb:1   # RTX PRO 6000 Blackwell, 96 GB (julia/fortran): same card for every run
#SBATCH --exclusive              # whole node: the CPU side uses all cores, no neighbours on the memory bus
#SBATCH --mem=0                  # all node memory (32 qubits = 64 GiB state on the host too)
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#
# 1 GPU vs 1 full CPU node, on the same node: QFT, Grover and random circuits,
# REPEATS runs each, fixed seeds.
#   GPU: qsim_cuda (built here for this card)
#   CPU: qsim built with OpenMP, one thread per physical core, pinned
#
#   sbatch slurm/cuda.sbatch
#   GPU_QUBITS="20 24" CPU_QUBITS="20 24" REPEATS=1 sbatch slurm/cuda.sbatch   # quick test
#   RUN_CPU=0 sbatch slurm/cuda.sbatch                                        # GPU only
#
# Output: results/cuda/<jobid>/{meta.txt, runs.csv, speedup.csv, raw/}
#   runs.csv:    one row per run (total_s = gate time only, h2d_s/d2h_s = GPU copies)
#   speedup.csv: median CPU time / median GPU time per circuit and size
#   raw/:        stdout and per-gate timing CSVs of the GPU (GB/s per gate)

set -euo pipefail

GPU_QUBITS=${GPU_QUBITS:-"20 22 24 26 28 30 31 32"}   # 32 qubits = 64 GiB, fits in 96 GB
CPU_QUBITS=${CPU_QUBITS:-"20 22 24 26 28 30"}         # the CPU side is much slower; 30 is enough to compare
KINDS=${KINDS:-"qft grover random"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-1}
RANDOM_DEPTH=${RANDOM_DEPTH:-10}
GROVER_ITERS=${GROVER_ITERS:-1}   # same as slurm/baseline.sbatch: 1 iteration is enough for time per gate
RUN_CPU=${RUN_CPU:-1}

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
source env/modules.sh
module load hpc-course-cuda/12.8.1

# Load every kernel when the GPU context starts, so the first gate of a run is
# not charged for lazy kernel loading (seen as a ~1 ms first gate in validation).
export CUDA_MODULE_LOADING=EAGER

JOBID=${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}
OUT="results/cuda/$JOBID"
mkdir -p "$OUT/raw" data/cuda

# ---------------------------------------------------------------- build
source slurm/lib/build_cuda.sh             # GPU: sets CUDA_ARCH, CUDA_BUILD, QSIM_CUDA_BIN

PHYS_CORES=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
if [ "$RUN_CPU" = 1 ]; then
  echo "--- building OpenMP qsim on $(hostname)"
  cmake -S . -B build-omp -DQSIM_OPENMP=ON -DQSIM_NATIVE=ON -DQSIM_TESTS=OFF >/dev/null
  cmake --build build-omp -j "$PHYS_CORES" --target qsim >/dev/null
fi

# ---------------------------------------------------------------- metadata
{
  echo "job_id:        $JOBID"
  echo "date:          $(date -Iseconds)"
  echo "host:          $(hostname)"
  echo "cpu:           $(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
  echo "physical_cores: $PHYS_CORES"
  echo "git_commit:    $(git rev-parse --verify -q HEAD || echo unknown)"
  echo "git_dirty:     $(git status --porcelain 2>/dev/null | grep -q . && echo yes || echo no)"
  echo "cuda_arch:     sm_$CUDA_ARCH"
  echo "gpu_binary:    $QSIM_CUDA_BIN"
  echo "cpu_binary:    build-omp/qsim (OpenMP, -march=native)"
  echo "gpu_qubits:    $GPU_QUBITS"
  echo "cpu_qubits:    $CPU_QUBITS"
  echo "kinds:         $KINDS"
  echo "repeats:       $REPEATS"
  echo "seed:          $SEED"
  echo "random_depth:  $RANDOM_DEPTH"
  echo "grover_iters:  $GROVER_ITERS"
  echo "CUDA_MODULE_LOADING=$CUDA_MODULE_LOADING"
  echo "--- module list"
  module list 2>&1
  echo "--- nvidia-smi"
  cuda-exec nvidia-smi
  echo "--- commands"
} > "$OUT/meta.txt"

echo "device,kind,qubits,repeat,seed,gates,total_s,mean_gate_s,h2d_s,d2h_s,norm_err,threads,circuit" > "$OUT/runs.csv"

# make_circuit KIND N -> prints the circuit path (generated once, reused by GPU and CPU)
make_circuit() {
  local kind=$1 n=$2 circ="data/cuda/${1}_${2}_s${SEED}.txt"
  if [ ! -f "$circ" ]; then
    case $kind in
      qft)    python3 src/tools/gen_circuits.py qft -n "$n" --seed "$SEED" -o "$circ" ;;
      grover) python3 src/tools/gen_circuits.py grover -n "$n" --seed "$SEED" --iterations "$GROVER_ITERS" -o "$circ" ;;
      random) python3 src/tools/gen_circuits.py random -n "$n" --seed "$SEED" --depth "$RANDOM_DEPTH" -o "$circ" ;;
      *) echo "unknown kind $kind" >&2; exit 1 ;;
    esac >/dev/null
  fi
  echo "$circ"
}

# run_one DEVICE THREADS KIND N REPEAT CMD... -> runs CMD, appends one row to runs.csv
run_one() {
  local device=$1 threads=$2 kind=$3 n=$4 r=$5; shift 5
  echo "$*" >> "$OUT/meta.txt"
  local line
  line=$("$@" | tee -a "$OUT/raw/stdout_${device}.txt" | grep '^SUMMARY')
  get() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<<"$line"; }
  echo "$device,$kind,$n,$r,$SEED,$(get gates_run),$(get total_s),$(get mean_gate_s),$(get h2d_s),$(get d2h_s),$(get norm_err),$threads,${@: -1}" \
    >> "$OUT/runs.csv"
  echo "[$(date +%T)] $device $kind n=$n rep=$r total_s=$(get total_s)"
}

# ---------------------------------------------------------------- GPU
for kind in $KINDS; do
  for n in $GPU_QUBITS; do
    circ=$(make_circuit "$kind" "$n")
    # One run with per-gate timing (GB/s per gate); not counted in runs.csv.
    "$QSIM_CUDA_BIN" "$circ" --quiet --timing "$OUT/raw/gpu_${kind}_${n}_gates.csv" >/dev/null
    # Timed repeats without per-gate syncs, so total_s is pure back-to-back kernel time.
    for r in $(seq 1 "$REPEATS"); do
      run_one gpu 0 "$kind" "$n" "$r" "$QSIM_CUDA_BIN" --quiet --no-gate-timing "$circ"
    done
  done
done

# ---------------------------------------------------------------- CPU (whole node)
if [ "$RUN_CPU" = 1 ]; then
  export OMP_NUM_THREADS=$PHYS_CORES OMP_PROC_BIND=spread OMP_PLACES=cores
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES" >> "$OUT/meta.txt"
  for kind in $KINDS; do
    for n in $CPU_QUBITS; do
      circ=$(make_circuit "$kind" "$n")
      for r in $(seq 1 "$REPEATS"); do
        run_one cpu "$OMP_NUM_THREADS" "$kind" "$n" "$r" build-omp/qsim --quiet --no-gate-timing "$circ"
      done
    done
  done
fi

# ---------------------------------------------------------------- speedup table
python3 - "$OUT" <<'EOF'
import csv, statistics, sys
from collections import defaultdict
out = sys.argv[1]
t = defaultdict(list)
for row in csv.DictReader(open(f"{out}/runs.csv")):
    t[(row["kind"], int(row["qubits"]), row["device"])].append(float(row["total_s"]))
keys = sorted({(k, n) for k, n, _ in t})
with open(f"{out}/speedup.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "qubits", "gpu_median_s", "cpu_median_s", "speedup"])
    print(f"{'kind':8}{'qubits':>7}{'gpu_s':>12}{'cpu_s':>12}{'speedup':>9}")
    for k, n in keys:
        g = statistics.median(t[(k, n, "gpu")]) if t[(k, n, "gpu")] else None
        c = statistics.median(t[(k, n, "cpu")]) if t[(k, n, "cpu")] else None
        s = c / g if g and c else None
        w.writerow([k, n, g or "", c or "", f"{s:.2f}" if s else ""])
        gs = f"{g:12.4g}" if g is not None else "-".rjust(12)
        cs = f"{c:12.4g}" if c is not None else "-".rjust(12)
        ss = f"{s:9.1f}" if s else "-".rjust(9)
        print(f"{k:8}{n:7d}{gs}{cs}{ss}")
EOF

echo "done: $OUT"
